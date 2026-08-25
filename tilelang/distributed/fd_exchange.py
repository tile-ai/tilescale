"""Cross-process file-descriptor exchange over Unix domain sockets.

CUDA POSIX-FD shareable handles (CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
are plain file descriptors. Unlike fabric handles they cannot be serialized
through torch.distributed object collectives; they must travel between
processes as SCM_RIGHTS ancillary data on a Unix domain socket. TileScale
process groups are single-node, so a Unix socket is always reachable.
"""

import os
import shutil
import socket
import tempfile
import time

import torch.distributed as dist

_CONNECT_TIMEOUT_S = 120.0


def _connect(path: str) -> socket.socket:
    deadline = time.monotonic() + _CONNECT_TIMEOUT_S
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(path)
            return sock
        except (FileNotFoundError, ConnectionRefusedError):
            sock.close()
            if time.monotonic() > deadline:
                raise
            time.sleep(0.01)


def broadcast_fd(fd: int, rank: int, num_ranks: int, group, src_global_rank: int) -> int:
    """Send ``fd`` from rank 0 to every other rank in ``group``.

    Returns ``fd`` unchanged on rank 0 and a received duplicate on other
    ranks. The caller owns the returned fd and must ``os.close`` it once the
    handle has been imported.
    """
    if num_ranks == 1:
        return fd

    server = None
    tmpdir = None
    if rank == 0:
        if fd < 0:
            raise ValueError(f"broadcast_fd requires a valid fd on rank 0, got {fd}")
        # Listen before publishing the path so no peer can connect too early.
        tmpdir = tempfile.mkdtemp(prefix="tilescale_fd_")
        path = os.path.join(tmpdir, "bcast.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen(num_ranks)
        obj = [path]
    else:
        obj = [None]

    dist.broadcast_object_list(obj, src=src_global_rank, group=group)
    path = obj[0]

    if rank == 0:
        try:
            for _ in range(num_ranks - 1):
                conn, _ = server.accept()
                try:
                    socket.send_fds(conn, [b"fd"], [fd])
                finally:
                    conn.close()
        finally:
            server.close()
            shutil.rmtree(tmpdir, ignore_errors=True)
        return fd

    conn = _connect(path)
    try:
        msg, fds, _, _ = socket.recv_fds(conn, 16, 1)
        if not msg or not fds:
            raise RuntimeError(f"rank {rank}: fd broadcast from rank 0 delivered no descriptor")
        return fds[0]
    finally:
        conn.close()


def exchange_fds(my_fd: int, rank: int, num_ranks: int, group, group_root_global_rank: int) -> list:
    """Full-mesh fd exchange: every rank receives every other rank's fd.

    Returns a list of ``num_ranks`` fds where entry ``rank`` is ``my_fd``
    itself and every other entry is a received duplicate owned by the caller.
    """
    if num_ranks == 1:
        return [my_fd]
    if my_fd < 0:
        raise ValueError(f"exchange_fds requires a valid fd, got {my_fd}")

    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix="tilescale_fd_")
        obj = [tmpdir]
    else:
        obj = [None]
    dist.broadcast_object_list(obj, src=group_root_global_rank, group=group)
    tmpdir = obj[0]

    path = os.path.join(tmpdir, f"r{rank}.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(num_ranks)

    fds = [-1] * num_ranks
    fds[rank] = my_fd
    try:
        # Everyone is listening after this barrier, so connects cannot race.
        dist.barrier(group=group)
        for peer in range(num_ranks):
            if peer == rank:
                continue
            conn = _connect(os.path.join(tmpdir, f"r{peer}.sock"))
            try:
                socket.send_fds(conn, [rank.to_bytes(4, "little")], [my_fd])
            finally:
                conn.close()
        for _ in range(num_ranks - 1):
            conn, _ = server.accept()
            try:
                msg, received, _, _ = socket.recv_fds(conn, 16, 1)
                if len(msg) < 4 or not received:
                    raise RuntimeError(f"rank {rank}: malformed fd-exchange message")
                sender = int.from_bytes(msg[:4], "little")
                fds[sender] = received[0]
            finally:
                conn.close()
    except BaseException:
        for peer, fd in enumerate(fds):
            if peer != rank and fd >= 0:
                os.close(fd)
        raise
    finally:
        server.close()
        # Peers may still be connecting to other sockets in the directory;
        # only remove after every rank has finished its sends and receives.
        dist.barrier(group=group)
        if rank == 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return fds
