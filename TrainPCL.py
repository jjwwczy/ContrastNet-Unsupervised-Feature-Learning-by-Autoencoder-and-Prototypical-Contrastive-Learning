import torch
import torch.nn as nn
import numpy as np
import faiss
from tqdm import tqdm
import time
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
class PairDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,Datapath1,Datapath2):
        # 1. Initialize file path or list of file names.
        self.DataList1=np.load(Datapath1)
        self.DataList2 = np.load(Datapath2)
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        index=index
        Data=(self.DataList1[index])
        # Data=Data.reshape(1,Data.shape[0])
        Data2=(self.DataList2[index])
        # Data2=Data2.reshape(1,Data2.shape[0])
        return torch.FloatTensor(Data), torch.FloatTensor(Data2), index
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.DataList1)
def run_kmeans(x, num_clusters, temperature):
    """
    Args:
        x: data to be clustered
    """
    print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': []}
    for seed, num_cluster in enumerate(num_clusters):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = False
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 5
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0
        index = faiss.GpuIndexFlatL2(res, d, cfg)
        clus.train(x, index)
        D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)
        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d
                # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax
        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  # clamp extreme values for stability
        density = temperature * density / density.mean()  # scale the mean to temperature
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)
        im2cluster = torch.LongTensor(im2cluster).cuda()
        density = torch.Tensor(density).cuda()
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)
    return results
def compute_features(eval_loader, model):
    print('Computing features...')
    model.eval()
    features = []
    for step, (x_i, x_j, index)in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            x_i = x_i.cuda().float()
            x_j = x_j.cuda().float()
            _, feat = model(x_j,is_eval=True)
        for num in range(len(feat)):  # i是AAE，j是CNN
            features.append(np.array(feat[num].cpu().detach().numpy()))
    return features
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
def adjust_learning_rate(original_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch>120:
        lr = original_lr * (0.1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch>160:
        lr = original_lr * (0.01)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # elif epoch>400:
    #     lr = original_lr * (0.001)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def train(train_loader, model, criterion, optimizer, epoch, num_clusters, cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst, acc_proto],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model=model.cuda()
    model.train()
    end = time.time()
    for i, (x_i,x_j, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        x_i = x_i.cuda()
        x_j = x_j.cuda()
        # compute output
        output, target, output_proto, target_proto = model(im_q=x_i, im_k=x_j,cluster_result=cluster_result, index=index)
        # InfoNCE loss
        loss = criterion(output, target)
        # ProtoNCE loss
        if output_proto is not None:
            loss_proto = 0
            for proto_out, proto_target in zip(output_proto, target_proto):
                loss_proto += criterion(proto_out, proto_target)
                accp = accuracy(proto_out, proto_target)[0]
                acc_proto.update(accp[0], x_i.size(0))
            # average loss across all sets of prototypes
            loss_proto /= len(num_clusters)
            loss = loss+loss_proto
        losses.update(loss.item(), x_i.size(0))
        acc = accuracy(output, target)[0]
        acc_inst.update(acc[0], x_i.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            progress.display(i)
def TrainContrast(PixelPath,PatchPath,batch_size,temperature,projection_dim=128,input_dim=128):
    from DefinedModels import Contrast, MoCo
    model=MoCo(projection_dim=projection_dim,input_dim=input_dim, r=640, m=0.999,T=temperature )
    print(model)
    train_data = PairDataset(PixelPath,PatchPath)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False,
                                              drop_last=False)
    criterion = nn.CrossEntropyLoss().cuda()
    SGD_lr=3e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=SGD_lr, momentum=0.9,weight_decay=0.001)
    num_clusters = [1000,1500,2500]
    for epoch in range(200):
        cluster_result = None
        adjust_learning_rate(SGD_lr, optimizer, epoch)
        if epoch >= 30:
            # compute momentum features for center-cropped images
            features = compute_features(test_loader, model)
            features=torch.tensor(features).float().cuda()
            # placeholder for clustering result
            cluster_result = {'im2cluster': [], 'centroids': [], 'density': []}
            for num_cluster in num_clusters:
                cluster_result['im2cluster'].append(torch.zeros(len(train_data), dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster), projection_dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())
            features[torch.norm(features, dim=1) > 1.5] /= 2  # account for the few samples that are computed twice
            features = features.cpu().numpy()
            cluster_result = run_kmeans(features, num_clusters, temperature)
        train(train_loader, model, criterion, optimizer, epoch, num_clusters, cluster_result)
        if (epoch%20==0):
            torch.save(model.state_dict(), './models/contrast.pth')
def ContrastPredict(PixelPath,PatchPath,batch_size, temperature,projection_dim,input_dim):
    from DefinedModels import Contrast, MoCo
    from Preprocess import feature_normalize2
    model = MoCo(projection_dim=projection_dim,input_dim=input_dim, r=640, m=0.999,
                 T=temperature)
    print(model)
    train_data = PairDataset(PixelPath, PatchPath)
    test_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False,
                                              drop_last=False)
    model.load_state_dict(torch.load('./models/contrast.pth'))
    model.eval()
    features = []
    for step, (x_i, x_j, index)in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            x_i = x_i.cuda().float()
            x_j = x_j.cuda().float()
            # feat, _ = model(x_j,is_eval=True)
            feat, _ = model(x_i,is_eval=True)
        for num in range(len(feat)):  # i是AAE，j是VAE
            features.append(np.array(feat[num].cpu().detach().numpy()))
    features=feature_normalize2(features)

    return features