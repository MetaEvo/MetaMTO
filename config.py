import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_state",type=int,default=5)
    parser.add_argument("--dim",type=int,default=50)
    parser.add_argument("--task_cnt",type=int,default=10)
    parser.add_argument("--episode",type=int,default=126)
    parser.add_argument("--difficulty", type=str, default="S")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--load_name", type=str, default=None)

    args = parser.parse_args()

    return args

