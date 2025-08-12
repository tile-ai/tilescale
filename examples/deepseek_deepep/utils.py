from dataclasses import dataclass
import torch
from typing import Union, Tuple


num_sms: int = 20

@dataclass
class Config:
    num_sms : int
    num_max_nvl_chunked_send_tokens : int
    num_max_nvl_chunked_recv_tokens : int
    num_max_rdma_chunked_send_tokens : int
    num_max_rdma_chunked_recv_tokens : int
    
    @staticmethod
    def get_dispatch_config(num_ranks: int) -> 'Config':
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(num_sms, 24, 256, 6, 128),
            4: Config(num_sms, 6, 256, 6, 128),
            8: Config(num_sms, 6, 256, 6, 128),
            16: Config(num_sms, 36, 288, 20, 128),
            24: Config(num_sms, 8, 288, 32, 128),
            32: Config(num_sms, 32, 288, 32, 128),
            64: Config(num_sms, 20, 288, 28, 128),
            128: Config(num_sms, 20, 560, 32, 128),
            144: Config(num_sms, 32, 720, 12, 128),
            160: Config(num_sms, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]
    
    
    @staticmethod
    def get_combine_config(num_ranks: int) -> 'Config':
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(num_sms, 10, 256, 6, 128),
            4: Config(num_sms, 9, 256, 6, 128),
            8: Config(num_sms, 4, 256, 6, 128),
            16: Config(num_sms, 4, 288, 12, 128),
            24: Config(num_sms, 1, 288, 8, 128),
            32: Config(num_sms, 1, 288, 8, 128),
            64: Config(num_sms, 1, 288, 20, 128),
            128: Config(num_sms, 1, 560, 12, 128),
            144: Config(num_sms, 2, 720, 8, 128),
            160: Config(num_sms, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]
    
    
def unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
    bias_0, bias_1 = None, None
    if isinstance(bias, torch.Tensor):
        bias_0 = bias
    elif isinstance(bias, tuple):
        assert len(bias) == 2
        bias_0, bias_1 = bias
    return bias_0, bias_1


