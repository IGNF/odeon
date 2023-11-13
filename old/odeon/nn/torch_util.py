import timeit
import torch


def get_num_thread_for_machine():

    threads = [1] + [t for t in range(2, 49, 2)]
    num_thread = 1
    best_time = None

    for t in threads:

        torch.set_num_threads(t)
        r = timeit.timeit(setup="import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)",
                          stmt="torch.mm(x, y)",
                          number=100)
        best_time = min(best_time, r) if best_time is not None else best_time
        num_thread = t if best_time >= r else num_thread

    return num_thread
