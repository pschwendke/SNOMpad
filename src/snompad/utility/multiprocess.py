# multiprocess manager to distribute discrete function calls onto different CPU cores

def process_manager(func, args: dict, collector: dict, identifier: int, cores: int = None):
    if cores is not None:
        raise NotImplementedError
    returns = func(**args)
    collector[identifier] = returns
