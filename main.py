from utils import train
from config import get_config

if __name__ == '__main__':
    config = get_config()
    train(250, config)
