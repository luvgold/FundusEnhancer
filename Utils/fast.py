from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import torch

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class DataPrefetcher():
    def __init__(self, loader,device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device=device
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    #################################### Temp ###############################
    # ----改造前----
    # for iter_id, batch in enumerate(data_loader):
    #     if iter_id >= num_iters:
    #         break
    #     for k in batch:
    #         if k != 'meta':
    #             batch[k] = batch[k].to(device=opt.device, non_blocking=True)
    #     run_step()
    #
    # # ----改造后----
    # prefetcher = DataPrefetcher(data_loader, opt)
    # batch = prefetcher.next()
    # iter_id = 0
    # while batch is not None:
    #     iter_id += 1
    #     if iter_id >= num_iters:
    #         break
    #     run_step()
    #     batch = prefetcher.next()
    #################################### Temp ###############################


class img_data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

