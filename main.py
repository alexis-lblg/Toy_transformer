from data import Data 
import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    a = Data()
    a.run()