import argparse
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import KFold, GridSearchCV
import json
import tqdm
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

# 设置数据集路径
dataset_paths = []


def load_per_file(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        prompt_ids, response_ids = data["prompt_ids"][0], data["response_ids"][0]
        prompt_length, response_length = len(prompt_ids), len(response_ids)
        relevance = data["relevance"]
        source2dst_relevance = [x[:prompt_length] for x in relevance]
        source_relevance = list(np.array(source2dst_relevance).mean(axis=0))
        dst_relevance = list(np.array(source2dst_relevance).mean(axis=-1))
        label = data["hallucination"]

    return source2dst_relevance, source_relevance, dst_relevance, label


def pad1D(array, left=False, N=300):
    # 获取原始数组的长度
    K = len(array)

    # 计算需要补零的数量
    padding_length = N - K

    # 如果原始数组已经大于或等于1024，就不需要补零
    if padding_length <= 0:
        return array[:N]

    # 创建补零后的数组
    padded_array = np.pad(array, (padding_length, 0) if left else (0, padding_length), mode='constant',
                          constant_values=0)

    return padded_array


def pad2D(matrix, N=300, K=300, top=False, left=False):
    # 获取原始矩阵的形状
    original_shape = matrix.shape

    # 截取
    if original_shape[0] >= N:
        truncate_matrix = matrix[:N, :]
    else:
        truncate_matrix = matrix
    if original_shape[1] >= K:
        truncate_matrix = truncate_matrix[:, :K]
    else:
        truncate_matrix = truncate_matrix

    shape = truncate_matrix.shape
    # 计算需要补零的数量
    pad_width = ((N - shape[0], 0) if top else (0, N - shape[0]), (K - shape[1], 0) if left else (0, K - shape[1]))

    # 创建补零后的矩阵
    padded_matrix = np.pad(truncate_matrix, pad_width, mode='constant', constant_values=(0))

    return padded_matrix


def adaptive_pca(X, variance_threshold=0.95):
    # 确保输入是 PyTorch 的 Tensor
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)

    # 数据中心化
    mean = X.mean(dim=0)
    X_centered = X - mean

    # 计算 SVD
    U, S, Vt = torch.linalg.svd(X_centered)

    # 计算累计方差
    total_variance = torch.sum(S ** 2)
    cumulative_variance = torch.cumsum(S ** 2, dim=0) / total_variance

    # 寻找达到阈值的最小主成分数量
    n_components = (cumulative_variance >= variance_threshold).nonzero(as_tuple=True)[0][0].item() + 1

    # 选择前 n_components 个最大的奇异值对应的右奇异向量
    selected_components = Vt[-n_components:]

    # 获取对应奇异值的下标
    selected_indices = torch.arange(len(S) - n_components, len(S))

    # 对原始数据进行变换
    transformed_data = torch.mm(X_centered, selected_components.transpose(0, 1))

    return transformed_data, selected_components, n_components, cumulative_variance, selected_indices


def average_segments(arr, new_length):
    # 确定原数组长度
    old_length = len(arr)

    # 计算比例因子
    ratio = old_length / new_length

    # 初始化新数组
    new_arr = []

    # 遍历并计算每个区间的均值
    for i in range(new_length):
        start_index = int(i * ratio)
        end_index = int((i + 1) * ratio)

        # 计算均值
        segment_sum = sum(arr[start_index:end_index])
        segment_avg = segment_sum / (end_index - start_index) if end_index > start_index else 0

        # 添加到新数组
        new_arr.append(segment_avg)

    return new_arr


def average_segments_2d(arr_2d, new_rows, new_cols):
    # 获取二维数组的行数和列数
    rows, cols = len(arr_2d), len(arr_2d[0])

    # 计算行和列的比例因子
    row_ratio = rows / new_rows
    col_ratio = cols / new_cols

    # 初始化新的二维数组
    new_arr_2d = []

    # 遍历新行
    for i in range(new_rows):
        new_row = []
        # 计算当前行的起始和结束索引
        start_row_index = int(i * row_ratio)
        end_row_index = int((i + 1) * row_ratio)

        # 遍历新列
        for j in range(new_cols):
            # 计算当前列的起始和结束索引
            start_col_index = int(j * col_ratio)
            end_col_index = int((j + 1) * col_ratio)

            # 计算均值
            segment_sum = 0
            count = 0
            for r in range(start_row_index, end_row_index):
                for c in range(start_col_index, end_col_index):
                    if r < rows and c < cols:  # 确保索引有效
                        segment_sum += arr_2d[r][c]
                        count += 1

            segment_avg = segment_sum / count if count > 0 else 0

            # 添加到新行
            new_row.append(segment_avg)

        # 添加新行到新数组
        new_arr_2d.append(new_row)

    return np.array(new_arr_2d)


def get_xs_ys(classifier="SVM", dataset_path=""):
    xs, ys = [], []

    for fp in os.listdir(dataset_path):
        if fp.endswith('.json'):
            sd_relevance, s_relevance, d_relevance, label = load_per_file(os.path.join(dataset_path, fp))
            if classifier == "SVM" or classifier == "RF" or classifier == "MLP":
                xs.append(np.array(average_segments(d_relevance, 235)).astype(np.float32))
            elif classifier == "LSTM":
                xs.append(average_segments_2d(sd_relevance, 300, 800).astype(np.float32))
            elif classifier == "Threshold":
                xs.append(np.sum(sd_relevance) / (len(sd_relevance) * len(sd_relevance[0])))
            elif classifier == "Threshold_source":
                xs.append(sum(normalize(average_segments(s_relevance, 110)[5:-5])) / 100)
            elif classifier == "Threshold_response":
                xs.append(sum(normalize(average_segments(d_relevance, 110)[5:-5])) / 100)
            else:
                raise Exception("Classifier not supported")
            ys.append(label)
    return xs, ys


# l=300 avg accuracy=0.6076090857816746,avg precision=0.5134907439421803,avg recall=0.5379332976636346, avg f1=0.5242036972972214
# l=270 avg accuracy=0.6268779162180177,avg precision=0.5390415052075668,avg recall=0.5416565008025682, avg f1=0.538454498639229
# l=260 avg accuracy=0.6106701533097472,avg precision=0.5133453074382747,avg recall=0.5347858034599607, avg f1=0.523073886041771
# l=250 avg accuracy=0.6258985797056863,avg precision=0.5327809171826627,avg recall=0.5536543606206528, avg f1=0.542437823114603
# gamma=1,c=300 avg accuracy=0.6258780700405066,avg precision=0.5341066565062162,avg recall=0.5491599785981809, avg f1=0.5404082642021858
# gamma=1,c=400 avg accuracy=0.6339691329539046,avg precision=0.5436059999849303,avg recall=0.5616543606206527, avg f1=0.5515138969527795
# gamma=1,c=500 avg accuracy=0.6299287289135005,avg precision=0.5376454601318242,avg recall=0.5581237738541109, avg f1=0.5469555141496033
# gamma=1,c=380 avg accuracy=0.6349792339640056,avg precision=0.5443760366533431,avg recall=0.5665432495095416, avg f1=0.5542559268359606
# gamma=1,c=370 avg accuracy=0.6359944623904015,avg precision=0.545640715966803,avg recall=0.5665432495095416, avg f1=0.5549418651015168
# gamma=1,c=360 avg accuracy=0.6349843613803005,avg precision=0.5444358966896945,avg recall=0.563876582842875, avg f1=0.5530789560974565
# gamma=1,c=350 avg accuracy=0.6349843613803005,avg precision=0.5447680837227591,avg recall=0.563876582842875, avg f1=0.5531269161423691
# gamma=1,c=410 avg accuracy=0.6339691329539046,avg precision=0.5436059999849303,avg recall=0.5616543606206527, avg f1=0.5515138969527795
# gamma=2,c=400 avg accuracy=0.6238886325180741,avg precision=0.5303199090713665,avg recall=0.5771328696272516, avg f1=0.5515578637223485


# l=240 avg accuracy=0.6137466030867047,avg precision=0.5204715424426437,avg recall=0.5501394685214909, avg f1=0.5326910248734886
# l=230 avg accuracy=0.6177870071271087,avg precision=0.5271836297169408,avg recall=0.5108378812199037, avg f1=0.517626583588487
# l=220 avg accuracy=0.6238578680203045,avg precision=0.5321814490419141,avg recall=0.5291235955056179, avg f1=0.5295658119399451
def get_xs_ys_from_multi_path(classifier="SVM", dataset_paths=[]):
    xs, ys = [], []
    for dataset_path in dataset_paths:
        _xs, _ys = get_xs_ys(classifier, dataset_path)
        xs.extend(_xs)
        ys.extend(_ys)
    return np.array(xs), np.array(ys)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        # out = self.fc(out[:, -1, :])
        out = self.fc(h_n[-1, :, :]).reshape(-1, output_size)
        out = torch.softmax(out, dim=1)
        return out


input_size = 800  # 输入特征维度
batch_size = 64  # 批次大小
hidden_size = 256  # LSTM 隐藏层大小
num_layers = 2  # LSTM 层数
output_size = 2  # 输出类别数
threshold = [0.8, 0.2]
relevance_threshold = 0.53

# Response llama-7b
# rt=0.4 avg accuracy=0.5490180997795211,avg precision=0.6982881408757986,avg recall=0.21853638282000837, avg f1=0.33155374394596804
# rt=0.41 avg accuracy=0.5560836794339333,avg precision=0.6832682794885547,avg recall=0.26029763753670976, avg f1=0.37653889893489767
# rt=0.42 avg accuracy=0.5722606778444341,avg precision=0.6841994478951001,avg recall=0.3196093278907077, avg f1=0.435261714611402
# rt=0.43 avg accuracy=0.5884633133364098,avg precision=0.6738687125600157,avg recall=0.3915966735378851, avg f1=0.4950462792030283
# rt=0.44 avg accuracy=0.5833974260370199,avg precision=0.6372832988607636,avg recall=0.4425186089427607, avg f1=0.5222296351627733
# rt=0.45 avg accuracy=0.59349843613803,avg precision=0.6306517578450291,avg recall=0.5068822792548711, avg f1=0.5616871496906665
# rt=0.46 avg accuracy=0.6036302107368098,avg precision=0.6259744121857523,avg recall=0.5653724756089862, avg f1=0.59364857015612
# rt=0.47 avg accuracy=0.601584371635133,avg precision=0.6100459090537165,avg recall=0.6185414494919367, avg f1=0.6137772956174008
# rt=0.48 avg accuracy=0.602599600061529,avg precision=0.5972028531759971,avg recall=0.6949143275596903, avg f1=0.6422562792350218
# rt=0.49 avg accuracy=0.5985643234374198,avg precision=0.5849082364938188,avg recall=0.7520780089464075, avg f1=0.657716814168479
# rt=0.5 avg accuracy=0.601584371635133,avg precision=0.5805947895033241,avg recall=0.8142349181638948, avg f1=0.6772254702520124
# rt=0.51 avg accuracy=0.5975747320924987,avg precision=0.5735465057818,avg recall=0.8534162857604543, avg f1=0.68553770414794
# rt=0.52 avg accuracy=0.5914987437830077,avg precision=0.5654947144960969,avg recall=0.9006008052877388, avg f1=0.6942255149807774
# rt=0.53 avg accuracy=0.5824129621083936,avg precision=0.5574250007283201,avg recall=0.925809652401888, avg f1=0.6952911616034427
# rt=0.54 avg accuracy=0.5753371276213916,avg precision=0.5512384302906406,avg recall=0.9487631255454353, avg f1=0.6968558780956918
# rt=0.55 avg accuracy=0.5601548479721068,avg precision=0.5410549360680834,avg recall=0.9704689194039199, avg f1=0.6943416705226622
# rt=0.56 avg accuracy=0.5500538378710967,avg precision=0.5346914691382842,avg recall=0.9842300476212025, avg f1=0.6925150401456551
# rt=0.57 avg accuracy=0.5369173973234886,avg precision=0.5273065581100633,avg recall=0.9860818994730544, avg f1=0.6867469973392046
# rt=0.58 avg accuracy=0.5328769932830847,avg precision=0.524976111216073,avg recall=0.9900034681005053, avg f1=0.6857154450849732
# rt=0.59 avg accuracy=0.5308567912628827,avg precision=0.5238463739592353,avg recall=0.9942587872494416, avg f1=0.6857194257414114
# rt=0.6 avg accuracy=0.5288365892426806,avg precision=0.5226761847686215,avg recall=0.9962789892696435, avg f1=0.6852267716335566



# Source llama-7b
# rt=0.4 avg accuracy=0.5268317694713633,avg precision=0.5222741345341965,avg recall=0.9641746163005214, avg f1=0.6771393626961919
# rt=0.39 avg accuracy=0.5268368968876583,avg precision=0.522409626654282,avg recall=0.95632157609454, avg f1=0.6753152846020398
# rt=0.38 avg accuracy=0.5298671999179614,avg precision=0.5241991578330124,avg recall=0.9482363137533737, avg f1=0.6748337027903417
# rt=0.37 avg accuracy=0.5268368968876583,avg precision=0.5228627907316433,avg recall=0.9364473477085037, avg f1=0.6707026948719512
# rt=0.36 avg accuracy=0.5339076039583655,avg precision=0.5270177455582878,avg recall=0.9247673141255073, avg f1=0.6711725474073157
# rt=0.35 avg accuracy=0.5318925293544583,avg precision=0.5268600012834154,avg recall=0.8973331602827586, avg f1=0.6635421679095433
# rt=0.34 avg accuracy=0.5450238424857715,avg precision=0.5354888650449476,avg recall=0.882004959470366, avg f1=0.6660538942769875
# rt=0.33 avg accuracy=0.5409680561964827,avg precision=0.534508978506713,avg recall=0.8573913133888805, avg f1=0.6578955652699537
# rt=0.32 avg accuracy=0.5510895759626724,avg precision=0.5422752452937998,avg recall=0.8331077728969956, avg f1=0.6564054658399485
# rt=0.31 avg accuracy=0.5642311439265754,avg precision=0.5522449947261977,avg recall=0.8130968585806994, avg f1=0.6574736819749532
# rt=0.30 avg accuracy=0.5561349535968825,avg precision=0.5498503587774464,avg recall=0.7556038809328551, avg f1=0.6363013768310649
# rt=0.29 avg accuracy=0.55307388606881,avg precision=0.549774558377778,avg recall=0.722510329374634, avg f1=0.6241679048473832
# rt=0.28 avg accuracy=0.55306363123622,avg precision=0.5528232947581155,avg recall=0.6853327881885776, avg f1=0.6116134781195331
# rt=0.27 avg accuracy=0.5570886530277392,avg precision=0.5598238006854132,avg recall=0.6485835727744422, avg f1=0.6004311080999292
# rt=0.26 avg accuracy=0.5540532225811414,avg precision=0.5622389217172805,avg recall=0.6016718816170233, avg f1=0.580694475536264
# rt=0.25 avg accuracy=0.5479977439368302,avg precision=0.5647868369254507,avg recall=0.528576758268078, avg f1=0.5453137021795655
# rt=0.24 avg accuracy=0.5378813515869354,avg precision=0.5611498221206559,avg recall=0.4639606385164501, avg f1=0.5073588771400199
# rt=0.23 avg accuracy=0.5348766856381069,avg precision=0.5702788160870352,avg recall=0.3993716066540742, avg f1=0.4691443011616155
# rt=0.22 avg accuracy=0.530826026765113,avg precision=0.5742755900013965,avg recall=0.3523930937146055, avg f1=0.4359509744072449
# rt=0.21 avg accuracy=0.5166435932933394,avg precision=0.5629703023576498,avg recall=0.2722060750361313, avg f1=0.36650871409632896
# rt=0.20 avg accuracy=0.5095626313900427,avg precision=0.5564483884128927,avg recall=0.23290180569256047, avg f1=0.3277912289792265

# Response llama-13b
# rt=0.3
# rt=0.31
# rt=0.32
# rt=0.33 avg accuracy=0.5925344818745835,avg precision=0.43015873015873013,avg recall=0.03695024077046549, avg f1=0.06799697110008972
# rt=0.34 avg accuracy=0.5905142798543814,avg precision=0.4216666666666667,avg recall=0.04675298733725701, avg f1=0.08402363178613359
# rt=0.35 avg accuracy=0.5965800133312824,avg precision=0.49757575757575767,avg recall=0.06451721062957019, avg f1=0.11394117433721393
# rt=0.36 avg accuracy=0.5975901143413834,avg precision=0.5156843156843157,avg recall=0.0822314963438559, avg f1=0.14138765353718624
# rt=0.37 avg accuracy=0.5935394554683895,avg precision=0.48723389355742297,avg recall=0.10308186195826645, avg f1=0.1698891773680487
# rt=0.38 avg accuracy=0.5935343280520945,avg precision=0.4871428571428571,avg recall=0.12526556090601035, avg f1=0.19917098002680284
# rt=0.39 avg accuracy=0.5945444290621955,avg precision=0.49406725208824154,avg recall=0.15285749955412878, avg f1=0.2327299942159171
# rt=0.4 avg accuracy=0.5905142798543814,avg precision=0.47767741935483865,avg recall=0.17529516675584092, avg f1=0.25595184850799824
# rt=0.41 avg accuracy=0.5935650925498641,avg precision=0.49200542005420056,avg recall=0.2353700731228821, avg f1=0.317391111584025
# rt=0.42 avg accuracy=0.6006357996205712,avg precision=0.5095676918110239,avg recall=0.31113893347601207, avg f1=0.38440971334761165
# rt=0.43 avg accuracy=0.6036507204019894,avg precision=0.5137692240602377,avg recall=0.3611549848403781, avg f1=0.4221942144012984
# rt=0.44 avg accuracy=0.5844229092960058,avg precision=0.48286270510800033,avg recall=0.4047718922775102, avg f1=0.4386270659113666
# rt=0.45 avg accuracy=0.5793724042455007,avg precision=0.4786007992793831,avg recall=0.47854752987337257, avg f1=0.4770379692095264
# rt=0.46 avg accuracy=0.5671999179613392,avg precision=0.4686755069381296,avg recall=0.5446634563937935, avg f1=0.5023356858417135
# rt=0.47 avg accuracy=0.5580987540378404,avg precision=0.46302919712354773,avg recall=0.6026295701801321, avg f1=0.52219766265196
# rt=0.48 avg accuracy=0.5469722606778444,avg precision=0.4577885555157188,avg recall=0.6719992866060283, avg f1=0.5433235179084437
# rt=0.49 avg accuracy=0.5348459211403374,avg precision=0.4527028614430189,avg recall=0.7391717495987159, avg f1=0.5604044133270407
# rt=0.5 avg accuracy=0.5186689227298364,avg precision=0.4455919027216397,avg recall=0.7908657035848047, avg f1=0.5686816121124176

# Source llama-13b
# rt=0.15 avg accuracy=0.5794031687432704,avg precision=0.36392156862745095,avg recall=0.07919743178170144, avg f1=0.12913158108102601
# rt=0.16 avg accuracy=0.5763523560477875,avg precision=0.39192307692307693,avg recall=0.10407276618512573, avg f1=0.1628173880265628
# rt=0.17 avg accuracy=0.5712864687483977,avg precision=0.4079785230249627,avg recall=0.1422179418583913, avg f1=0.20868031434579928
# rt=0.18 avg accuracy=0.5722965697584987,avg precision=0.4376628505632995,avg recall=0.20365115034777953, avg f1=0.27576364222975663
# rt=0.19 avg accuracy=0.5712915961646926,avg precision=0.44620957473898654,avg recall=0.25345175673265563, avg f1=0.3217579495392071
# rt=0.2 avg accuracy=0.5672255550428139,avg precision=0.44814253646237096,avg recall=0.3056286784376672, avg f1=0.36178951036827267
# rt=0.21 avg accuracy=0.5682612931343896,avg precision=0.4579683319835481,avg recall=0.37625468164794, avg f1=0.4113561567408034
# rt=0.22 avg accuracy=0.5490488642772907,avg precision=0.44049707602339183,avg recall=0.4311346531121812, avg f1=0.4340629852199627
# rt=0.23 avg accuracy=0.5308414090139978,avg precision=0.42694219051053317,avg recall=0.47180595683966475, avg f1=0.446417100933166
# rt=0.24 avg accuracy=0.5055581192637031,avg precision=0.41121426559837204,avg recall=0.516171214553237, avg f1=0.45572266127765937
# rt=0.25 avg accuracy=0.49239091421832537,avg precision=0.4088369247735525,avg recall=0.5733233458177278, avg f1=0.47555976688242146
# rt=0.26 avg accuracy=0.4792442188381275,avg precision=0.40815055396523603,avg recall=0.6374232209737828, avg f1=0.495294334689327
# rt=0.27 avg accuracy=0.4640721940214326,avg precision=0.4033566040980244,avg recall=0.6773461744248261, avg f1=0.5034226884226884
# rt=0.28 avg accuracy=0.45497103009793366,avg precision=0.401637577349251,avg recall=0.7126320670590334, avg f1=0.5120041319435058
# rt=0.29 avg accuracy=0.4499256524637235,avg precision=0.4039662173907056,avg recall=0.7620017834849296, avg f1=0.5265582590219557
# rt=0.3 avg accuracy=0.4489309337025073,avg precision=0.4077645701878061,avg recall=0.8093397538790799, avg f1=0.541087935936069
# rt=0.31 avg accuracy=0.4398092601138287,avg precision=0.4058526718235974,avg recall=0.8379611200285357, avg f1=0.5455683412477559
# rt=0.32 avg accuracy=0.4337383992206327,avg precision=0.4056383351469114,avg recall=0.8701879793115749, avg f1=0.5521402469965382
# rt=0.33 avg accuracy=0.43575860124083476,avg precision=0.4090477141904928,avg recall=0.8984283930800785, avg f1=0.5608237568566106
# rt=0.34 avg accuracy=0.42464236271342876,avg precision=0.4050566422210545,avg recall=0.9083966470483326, avg f1=0.5588557562873975
# rt=0.35 avg accuracy=0.4125262780085115,avg precision=0.40024074922380015,avg recall=0.9165871232388086, avg f1=0.5558846194913176

# avg accuracy=0.6835153566118033,avg precision=0.638831764589894,avg recall=0.4849195648296772, avg f1=0.543525958219623

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


def normalize(m):
    min_val = np.min(m)
    max_val = np.max(m)
    normalized_matrix = (m - min_val) / (max_val - min_val)
    return normalized_matrix


# 划分训练集和验证集
# X_train, X_val, y_train, y_val = train_test_split(xs, ys, test_size=0.2, random_state=42)
# 初始化 KFold 分割器
accuracy_score_list = []
precision_score_list = []
recall_score_list = []
f1_score_list = []


#llama-7b
# gamma=1,c=50 avg accuracy=0.6784853612264781,avg precision=0.679917173009274,avg recall=0.7117041913487236, avg f1=0.6946924993841556
# gamma=1,c=30 avg accuracy=0.6895913449212941,avg precision=0.6904227794252508,avg recall=0.7234451175458041, avg f1=0.705792661493596
# gamma=1.1,c=30,avg accuracy=0.691621801774086,avg precision=0.6935112791636695,avg recall=0.7213174579713361, avg f1=0.7064226957416259
def do_classification(classifier="SVM"):
    for xs, ys in [
        get_xs_ys_from_multi_path(classifier, dataset_paths=dataset_paths),
    ]:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        print(
            "----------------------------------------------------------------------------------------------------------")
        for train_index, test_index in kf.split(xs):
            # 根据索引分割训练集和测试集
            X_train, X_test = xs[train_index], xs[test_index]
            y_train, y_test = ys[train_index], ys[test_index]

            if classifier == "SVM":
                # 创建 SVM 分类器实例
                # 定义参数网格
                # param_grid = {
                #     'C': [  150,200,300,400, 500 ],
                #     'kernel': ['linear', 'rbf'],
                #     'gamma': ['scale', 'auto', 1, 2, 3,4,5]
                # }
                # svm_classifier = SVC()
                svm_classifier = SVC(gamma=1, C=30, kernel='rbf')
                # recall_scorer = make_scorer(recall_score)
                #
                # grid_search = GridSearchCV(svm_classifier, param_grid, scoring=recall_scorer)

                # 训练 SVM 模型
                svm_classifier.fit(X_train, y_train)
                # best_params = grid_search.best_params_
                # print(best_params)
                # 在训练集上预测
                y_train_pred = svm_classifier.predict(X_train)
                # 在验证集上进行预测
                y_pred = svm_classifier.predict(X_test)
            elif classifier == "RF":
                # 随机森林
                clf = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42)
                clf.fit(X_train, y_train)
                y_train_pred = clf.predict(X_train)
                y_pred = clf.predict(X_test)
            elif classifier == "MLP":
                clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001,
                                    max_iter=200,
                                    random_state=42)
                clf.fit(X_train, y_train)
                y_train_pred = clf.predict(X_train)
                y_pred = clf.predict(X_test)
            elif classifier.startswith("Threshold"):
                y_train_pred = np.array([0 if x > relevance_threshold else 1 for x in X_train])
                y_pred = np.array([0 if x > relevance_threshold else 1 for x in X_test])
            elif classifier == "LSTM":
                train_set = CustomDataset(X_train, y_train)
                val_set = CustomDataset(X_test, y_test)
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
                # 初始化模型
                model = LSTMClassifier(input_size, hidden_size, num_layers, output_size)

                # 定义损失函数和优化器
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

                # 训练循环
                num_epochs = 20
                best_acc, best_pre, best_recall, best_f1 = 0, 0, 0, 0
                for epoch in range(num_epochs):
                    total_loss = 0
                    for batch in tqdm.tqdm(train_loader):
                        X, y = batch
                        # 前向传播
                        outputs = model(X)

                        # 计算损失
                        loss = criterion(outputs, y)
                        total_loss += loss.item()

                        # 反向传播和优化
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

                    # 评估模型
                    with torch.no_grad():
                        labels, preds = [], []
                        for batch in val_loader:
                            X, y = batch
                            outputs = model(X)
                            outputs = (outputs > torch.tensor(threshold)).float()
                            preds.extend(torch.argmax(outputs, dim=1).tolist())
                            labels.extend(y.tolist())

                        # 计算准确率
                        accuracy = accuracy_score(labels, preds)
                        precision = precision_score(labels, preds)
                        recall = recall_score(labels, preds)
                        f1 = f1_score(labels, preds)
                        if accuracy + precision + recall + f1 > best_acc + best_pre + best_recall + best_f1:
                            best_acc = accuracy
                            best_pre = precision
                            best_recall = recall
                            best_f1 = f1

                        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
                        print(f"Precision: {precision * 100:.2f}%")
                        print(f"Recall: {recall * 100:.2f}%")
                        print(f"F1: {f1 * 100:.2f}%")
                accuracy_score_list.append(best_acc)
                precision_score_list.append(best_pre)
                recall_score_list.append(best_recall)
                f1_score_list.append(best_f1)
                continue

            accuracy_train, precision_train, recall_train = accuracy_score(y_train, y_train_pred), precision_score(
                y_train,
                y_train_pred), recall_score(
                y_train, y_train_pred)
            print(f"Training Accuracy: {accuracy_train * 100:.2f}%")
            print(f"Training Precision: {precision_train * 100:.2f}%")
            print(f"Training Recall: {recall_train * 100:.2f}%")

            accuracy_val, precision_val, recall_val, f1_val = accuracy_score(y_test, y_pred), precision_score(y_test,
                                                                                                              y_pred), recall_score(
                y_test, y_pred), f1_score(y_test, y_pred)
            print(f"Validation Accuracy: {accuracy_val * 100:.2f}%")
            print(f"Precision: {precision_val * 100:.2f}%")
            print(f"Recall: {recall_val * 100:.2f}%")
            print(f"F1: {f1_val * 100:.2f}%")
            accuracy_score_list.append(accuracy_val)
            precision_score_list.append(precision_val)
            recall_score_list.append(recall_val)
            f1_score_list.append(f1_val)


# avg accuracy=0.6410962416038558,avg precision=0.5619255211705785,avg recall=0.6528674870697342, avg f1=0.5909493240160002
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename files from rag-llm-prompt- to rag-prompt-llama-')

    parser.add_argument('--classifier',
                        choices=['LSTM', 'SVM', 'RF', 'MLP', "Threshold_source", "Threshold_response", "Threshold"],
                        required=True,
                        help='Classifier type (LSTM, SVM, random_forest, or MLP)', default='SVM')

    parser.add_argument('--datasets', type=str, default="/Users/tom/PycharmProjects/lrp-analysis/lrp_result_llama_7b",
                        help='lrp output paths, sep by ,')

    # 解析命令行参数
    args = parser.parse_args()
    dataset_paths = args.datasets.split(",")

    do_classification(args.classifier)
    print(
        f"avg accuracy={sum(accuracy_score_list) / len(accuracy_score_list)},avg precision={sum(precision_score_list) / len(precision_score_list)},avg recall={sum(recall_score_list) / len(recall_score_list)}, avg f1={sum(f1_score_list) / len(f1_score_list)}")
