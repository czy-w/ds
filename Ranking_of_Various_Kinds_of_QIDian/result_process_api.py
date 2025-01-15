import numpy as np 
import hashlib,os
from PIL import Image
from PIL.ExifTags import TAGS
import cv2

import functools
import time

def record_event(fn):  
    frequency_infer = 0
    def inner(*args, **kwargs):  
        nonlocal frequency_infer
        print(f"----------请注意，第{frequency_infer+1}次调用开始----------")  
        fn(*args, **kwargs)  
        frequency_infer += 1
        print(f"----------请注意，第{frequency_infer}次调用结束----------")      
    return inner 

@record_event
def inner():
    print("Hello, World!")


def log_execution(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Starting execution of {func.__name__} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        result = func(*args, **kwargs)
        print(f"Finished execution of {func.__name__} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return result
    return wrapper

@log_execution
def my_function():
    print("Hello, World!")

# my_function()



def measure_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to execute.")
        return result
    return wrapper

@measure_performance
def time_consuming_function():
    time.sleep(2)


def require_permission(permission_level):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 假设这里有获取当前用户权限的逻辑
            current_permission = "admin"  
            if current_permission >= permission_level:
                return func(*args, **kwargs)
            else:
                print("You don't have the required permission.")
        return wrapper
    return decorator

@require_permission("manager")
def restricted_function():
    print("This is a restricted function.")

# restricted_function()


def cache_result(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

@cache_result
def expensive_calculation(n):
    print(f"Calculating for {n}")
    return n * 2

# print(expensive_calculation(5))
# print(expensive_calculation(5))




def get_exif_data(fname):
    """Get embedded EXIF data from image file."""
    ret = {}
    try:
        img = Image.open(fname)
        if hasattr(img, '_getexif'):
            exifinfo = img._getexif()
            if exifinfo != None:
                for tag, value in exifinfo.items():
                    decoded = TAGS.get(tag, tag)
                    ret[decoded] = value
    except IOError:
        print('IOERROR ' + fname)
    return ret


def rot90(x1, y1, x2, y2, w_new, h_new):
    w_old = w_new
    h_old = h_new
    xmin = h_old - y1
    ymin = x1
    xmax = h_old - y2
    ymax = x2
    return min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)


def rot180(x1, y1, x2, y2, w_new, h_new):
    w_old = w_new
    h_old = h_new
    xmin = w_old - x1
    ymin = h_old - y1
    xmax = w_old - x2
    ymax = h_old - y2
    return min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)


def rot270(x1, y1, x2, y2, w_new, h_new):
    w_old = w_new
    h_old = h_new
    xmin = y1
    ymin = w_old - x1
    xmax = y2
    ymax = w_old - x2
    return min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)


def flip(x1, y1, x2, y2, w_new, h_new):
    w_old = w_new
    h_old = h_new
    xmin = w_old - x1
    ymin = y1
    xmax = w_old - x2
    ymax = y2
    return min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)


def transform(x1, y1, x2, y2, w_new, h_new, orientation):
    if orientation == 6:
        return rot270(x1, y1, x2, y2, w_new, h_new)
    elif orientation == 8:
        return rot90(x1, y1, x2, y2, w_new, h_new)
    elif orientation == 3:
        return rot180(x1, y1, x2, y2, w_new, h_new)
    elif orientation == 2:
        return flip(x1, y1, x2, y2, w_new, h_new)
    elif orientation == 5:
        x1, y1, x2, y2 = flip(x1, y1, x2, y2, w_new, h_new)
        return rot270(x1, y1, x2, y2, w_new, h_new)
    elif orientation == 7:
        x1, y1, x2, y2 = flip(x1, y1, x2, y2, w_new, h_new)
        return rot90(x1, y1, x2, y2, w_new, h_new)
    elif orientation == 4:
        x1, y1, x2, y2 = flip(x1, y1, x2, y2, w_new, h_new)
        return rot180(x1, y1, x2, y2, w_new, h_new)


def transform_results(results, w_new, h_new, orientation):
    for result in results:
        x1 = int(result[2])
        y1 = int(result[3])
        x2 = int(result[4])
        y2 = int(result[5])
        xmin, ymin, xmax, ymax = transform(
            x1, y1, x2, y2, w_new, h_new, orientation)
        result[2] = xmin
        result[3] = ymin
        result[4] = xmax
        result[5] = ymax



def py_cpu_nms(dets, thresh, iou_min_area=False, score_thresh = 0.0):
    """
    py_cpu_nms

    Args:
       dets (numpy((N, 5),float)): 
       thresh (float): iou thresh
       iou_min_area(bool) : (iou = inter / min_area) or  (iou = inter / union)
    Returns:
        keep (numpy(1,)): index to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if iou_min_area:
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
        inds_tmp = np.where(ovr > nms_thresh)[0]
        if np.size(inds_tmp) > 0:
            for ind in np.nditer(inds_tmp):
                differ = abs(scores[i] - scores[order[ind+1]])
                if differ <= score_thresh:
                    keep.append(order[ind+1])

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        
    return keep

def set_single_class_result(result_list, labels, class_name, np_ret):
    """
    删除一些仅仅被单个模型检出的结果

    Args:
        result_list：模型的输出结果（多个类）
        labels: 模型的标签
        class_name(str): 类名
        np_ret: new result
    Returns:

    """
    if (class_name not in labels) or (result_list is None):
        return
    idx = labels.index(class_name)
    if idx >= len(result_list):
        return
    result_list[idx] = np_ret
    
def find_single_class_result(result_list, labels, class_name):
    """
    在1个模型的输出结果中,找出特定的1个类的结果

    Args:
        result_list：模型的输出结果（多个类）
        labels: 模型的标签
        class_name(str): 类名
    Returns:
        single_class_result (numpy((N,5),dtype=np.float32))): 单个类别的结果或者None
    """
    if (class_name not in labels) or (result_list is None):
        return None
    idx = labels.index(class_name)
    if idx >= len(result_list):
        return None
    return result_list[idx]

def inter_class_nms(label_result_merged_dict, inter_class_nms_list, score_thresh=0.0):
    """
    类间nms

    Args:
        label_result_merged_dict: {label, 单类nms后的numpy结果}，做类间nms后，一些类别的结果会别抑制掉
        inter_class_nms_list:  [(label1, label2, lalbe3), (label1, label2)]，需要类间nms的类别
    Returns:
        无，
    """
    if len(inter_class_nms_list) < 2:
        return
    valid_labels = []
    result_list = []
    result_with_cls_list = []
    offset_list = [0, ]
    offset = 0
    index = 0

    for label in inter_class_nms_list:
        result = label_result_merged_dict.get(
            label, np.zeros((0, 5), dtype=np.float32))
        if 0 == result.shape[0]:
            continue
        np_cls = np.full((result.shape[0], 1), index, dtype=np.float32)
        result_with_cls = np.concatenate((result, np_cls), axis=1)
        valid_labels.append(label)
        result_list.append(result)
        result_with_cls_list.append(result_with_cls)
        offset += result.shape[0]
        offset_list.append(offset)
        index += 1
    if index < 2:
        return
    result_merge = np.concatenate(result_with_cls_list, axis=0)
    keep = py_cpu_nms(result_merge, nms_thresh, iou_min_area, score_thresh)
    keep_cls_list = []
    for cls_id in range(index):
        keep_cls_list.append([])
    for k in keep:
        cls_id = int(result_merge[k][5])
        keep_cls_list[cls_id].append(k-offset_list[cls_id])
    for cls_id in range(index):
        if len(keep_cls_list[cls_id]) > 0:
            label_result_merged_dict[valid_labels[cls_id]
                                     ] = result_list[cls_id][keep_cls_list[cls_id]]
        else:
            label_result_merged_dict[valid_labels[cls_id]] = np.zeros(
                (0, 5), dtype=np.float32)
            
def find_max_iou(dets, tl_x, tl_y, br_x, br_y):
    obj_area = (br_x - tl_x + 1) * (br_y - tl_y + 1)

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    xx1 = np.maximum(tl_x, x1)
    yy1 = np.maximum(tl_y, y1)
    xx2 = np.minimum(br_x, x2)
    yy2 = np.minimum(br_y, y2)

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / np.minimum(obj_area, areas)
    order = ovr.argsort()[::-1]
    idx = order[0]
    return ovr[idx]

def get_md5(img_path):
    with open(os.path.join(img_path), 'rb') as fd:
        f = fd.read()
        pmd5 = hashlib.md5(f)
    return pmd5.hexdigest()


def parase_md5_txt(md5_txt_path):
    with open(md5_txt_path, 'r') as f:
        md5_dict = {}
        md5_list = f.readlines()[1:]
        for item_str in md5_list:
            tem_md5_dict = {}
            item = item_str.strip().split(',')
            md5_key = item[1]
            md5_value = [(item[2], round(float(item[3]), 2), int(item[4]), int(item[5]), int(item[6]), int(item[7]))]
            # md5_value = [(item[2], 1.0, int(item[3]), int(item[4]), int(item[5]), int(item[6]))]
            tem_md5_dict[md5_key] = md5_value
            for key,value in tem_md5_dict.items():
                if key in md5_dict:
                    md5_dict[key].append(value[0])
                else:
                    md5_dict[key] = value
    return md5_dict

def convert_paddle_torch(results, label_conf_thresh_dict, classes):
    """
    paddle result:[[ 0.0000000e+00  4.5649819e-02  9.9363391e+02  3.5584277e+02
   1.3563317e+03  8.1965814e+02]
 [ 0.0000000e+00  2.4842652e-02 -8.4763360e-01 -3.1533689e+00
   5.1645287e+01  1.7325926e+02]
 [ 0.0000000e+00  2.0689759e-02  9.7201843e+02  3.2768472e+02
   1.3671877e+03  1.0606912e+03]
 ...
 [ 1.6000000e+01  2.3090398e-02  4.4990683e+02  6.4151379e+02
   5.2990302e+02  7.9283478e+02]
 [ 1.6000000e+01  2.1810135e-02  1.3307520e+03  9.6307220e+02
   1.6980045e+03  1.0804520e+03]
 [ 1.6000000e+01  2.0378290e-02  1.3481796e+03  8.7181561e+02
   1.9160466e+03  1.0798453e+03]]
    """
    np_list = [[] for _ in range(len(classes))]
    for result in results:
        id = int(result[0])
        score = round(float(result[1]),2)
        xmin = round(float(result[2]),2)
        ymin = round(float(result[3]),2)
        xmax = round(float(result[4]),2)      
        ymax = round(float(result[5]),2)
        new_list = [xmin, ymin, xmax, ymax, score]
        for i in range(0,len(np_list)):
            if id == i:
                ndarray1 = np.array(new_list,dtype='float32')
                np_list[i].append(ndarray1)
    results = []
    for i in np_list:
        ndarray1 = np.array(i,dtype='float32')
        results.append(ndarray1)

    if label_conf_thresh_dict is None:
        return results
    for idx, res in enumerate(results):
        label = classes[idx]
        if label in label_conf_thresh_dict:
            conf_thresh = label_conf_thresh_dict.get(label, 0.05)
            if res.shape != (0,):
                results[idx] = res[np.where(res[:, 4] >= conf_thresh)]
    return results  


def limit_output_size(result, area_size_range, result_list):
    item_area = (result[4] - result[2]) * (result[5] - result[3])
    if area_size_range[0] < item_area < area_size_range[1]:
        result_list.append(result)
    return result_list
    

def limit_output_num(result_list, max_output_num = -1):
    """
    Args:
       result_list (list): [[name,score,xmin,ymin,xmax,ymax], ]
       max_output_num(int): presents max output num

    Returns:
        output_result_list(list): [[name,score,xmin,ymin,xmax,ymax], ]
    """
    # sort list result and limit output num
    sorted_result_list = sorted(result_list, key = lambda x : x[1], reverse = True)
    # print(sorted_result_list)
    output_result_list = sorted_result_list
    # print("max_output_num--->", max_output_num)
    if max_output_num > 0:
        output_num = min(max_output_num, len(result_list))
        output_result_list = sorted_result_list[:output_num]

    elif max_output_num == -1:
        output_result_list = sorted_result_list
    # print("333output_result_list--->", output_result_list)
    return output_result_list



def limit_output_num_for_every_class(result_list, class_num_dict):
    """
    Args:
       result_list (list): [[name,score,xmin,ymin,xmax,ymax], ]
       class_num_dict(dict): {name1:num1, name2:num2,...}

    Returns:
        output_result_list(list): [[name,score,xmin,ymin,xmax,ymax], ]
    """
    if 0 == len(result_list) or 0 == len(class_num_dict):
        return result_list

    result_list_dict = dict()
    for result in result_list:
        name = result[0]
        if not name in result_list_dict.keys():
            result_list_dict[name] = list()
            result_list_dict[name].append(result)
        else:
            result_list_dict[name].append(result)

    # filter output class num and nonoutput class
    output_result_list = list()
    for name, result_list in result_list_dict.items():
        if name in class_num_dict:
            num = class_num_dict[name]
            every_class_output_result_list = limit_output_num(result_list, num)
            output_result_list.extend(every_class_output_result_list)
        # else:
        #     output_result_list.extend(result_list)
            
    # for name, num in class_num_dict.items():
    #     if name in result_list_dict.keys():
    #         result_list = result_list_dict[name]
    #         every_class_output_result_list = limit_output_num(result_list, num)
    #         output_result_list.extend(every_class_output_result_list)
    #     else:
    #         output_result_list.extend(result_list_dict.get(name, []))
    # print("444output_result_list--->", output_result_list)
    return output_result_list


def filter_reuslt_according_classlist(result_list, classlist):
    """ 
    Args:
        result_list(list): [[label, score, xmin, ymin, xmax, ymax],...]
        classlist:[name1, name2, ...]

    Returns:
        p_result_list(list): [[label, score, xmin, ymin, xmax, ymax],...]
    """ 
    # print(classlist)
    # print('before filter result_list',result_list)
    p_result_list = []
    if not result_list or (hasattr(result_list, '__len__') and len(result_list) == 0):
        return p_result_list
    
    if len(classlist) == 0 :
        return result_list
    
    for result in result_list:
        name = result[0]
        if name in classlist:
            p_result_list.append(result)

    # print('after filter result_list',p_result_list)    
    return p_result_list


def filter_reuslt_list(result_list, label_conf_thresh_dict):
    """ 
    Args:
        result_list(list): [[label, score, xmin, ymin, xmax, ymax],...]
        label_conf_thresh_dict(dict):{name1:conf_thresh1, name2:conf_thresh2, ...}

    Returns:
        p_result_list(list): [[label, score, xmin, ymin, xmax, ymax],...]
    """ 
    p_result_list = []
    if not result_list or (hasattr(result_list, '__len__') and len(result_list) == 0):
        return p_result_list
    
    if len(label_conf_thresh_dict) == 0 :
        return result_list
    
    for result in result_list:
        name = result[0]
        if name in label_conf_thresh_dict.keys():
            score = result[1]
            conf_thresh = label_conf_thresh_dict[name]
            if score > conf_thresh:
                p_result_list.append(result)
        else:
            p_result_list.append(result)

    return p_result_list 

def filter_box_based_on_label_conf_thresh_dict(result, id_to_class_list, label_conf_thresh_dict):
    """ 
    Args:
        result(np.ndarray): np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
        id_to_class_list(list): [name1, name2, ...]
        label_conf_thresh_dict(dict):{name1:conf_thresh1, name2:conf_thresh2, ...}

    Returns:
        result_list (list): [[code, score, x_min, y_min, x_max, y_max],...]
    """
    # print(result)
    result_list = list()
    if result.shape[0] == 0:
        return result_list

    # add other logical process 20221121
    max_label = str()
    max_score = 0
    max_xmin, max_ymin, max_xmax, max_ymax = 0, 0, 0, 0
    max_result = list()
    # add other logical process 20221121

    for i in range(0, result.shape[0]):
        idx = int(float(result[i][0]))
        label = id_to_class_list[idx]
        if label not in label_conf_thresh_dict.keys():
            continue
        conf_thresh = label_conf_thresh_dict[label]
        score = float(result[i][1])

        # add other logical process 20221121
        # score = 0.25
        if (len(result_list) == 0) and (score < conf_thresh) and (score > 0.2) and (score > max_score):
            max_label = label
            max_score = score
            max_xmin = int(float(result[i][2]))
            max_ymin = int(float(result[i][3]))
            max_xmax = int(float(result[i][4]))
            max_ymax = int(float(result[i][5]))
            max_result = [max_label, max_score,
                          max_xmin, max_ymin, max_xmax, max_ymax]
        # add other logical process 20221121

        if score < conf_thresh:
            continue

        result_list.append([label,
                            score,
                            int(float(result[i][2])),
                            int(float(result[i][3])),
                            int(float(result[i][4])),
                            int(float(result[i][5]))
                            ]
                           )

    # add other logical process 20221121
    if (len(result_list) == 0) and (len(max_result) == 6):
        result_list.append(max_result)
    # add other logical process 20221121

    return result_list


def py_cpu_nms_ori(dets, thresh, iou_min_area=False):
    """
    Args:
       dets (numpy((N, 5),float)): 
       thresh (float): iou thresh
       iou_min_area(bool) : (iou = inter / min_area) or  (iou = inter / union)

    Returns:
        keep (numpy(1,)): index to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if iou_min_area:
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_result_list_in_same_class_boxes(result_list, label_nms_thresh_dict, iou_min_area=False):
    """
    args:
        result_list(list): [[code, score, xmin, ymin, xmax, ymax], [code, score, xmin, ymin, xmax, ymax]]
        label_nms_thresh_dict(dict):{label1 : thresh1, label2 : thresh2,... }
        iou_min_area(bool) : (iou = inter / min_area) or  (iou = inter / union)

    return:
        nms_result_list(list): [[code, score, xmin, ymin, xmax, ymax], [code, score, xmin, ymin, xmax, ymax]]
    """
    # get effective dict result
    result_dict = dict()
    for result in result_list:
        name = result[0]
        score = result[1]
        x1 = result[2]
        y1 = result[3]
        x2 = result[4]
        y2 = result[5]
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            continue
        if not name in result_dict:
            result_dict[name] = list()
        result_dict[name].append([x1, y1, x2, y2, score])

    # get nms result list
    nms_result_list = list()
    for name, dets_list in result_dict.items():
        if not name in label_nms_thresh_dict.keys():
            continue
        nms_thresh = label_nms_thresh_dict[name]
        dets_np = np.array(dets_list)
        keep = py_cpu_nms(dets_np, nms_thresh, iou_min_area)
        for t in keep:
            nms_result_list.append([name, dets_list[t][4], dets_list[t]
                                   [0], dets_list[t][1], dets_list[t][2], dets_list[t][3]])

    return nms_result_list


def nms_result_list_in_different_class_boxes(result_list, inter_classes, inter_nms_thresh=0.7, iou_min_area=False, score_thresh=0.0):
    """
    Args:
       result_list(list) :[[name,score,xmin,ymin,xmax,ymax], ]
       inter_classes(tuple or list): (label1, label2,...)
       inter_nms_thresh(float):inter nms thresh
       iou_min_area(bool) : (iou = inter / min_area) or  (iou = inter / union)

    Returns:
        res_list_after_nms (list):[[name,score,xmin,ymin,xmax,ymax], ]
    """
    if result_list is None:
        return []

    if 0 == len(result_list):
        return result_list

    if len(inter_classes) == 0:
        return result_list

    res_list_nms = list()
    res_list_nms_data = list()
    res_list_after_nms_data = list()
    for temp in result_list:
        if temp[0] in inter_classes:
            res_list_nms.append([temp[2], temp[3], temp[4], temp[5], temp[1]])
            res_list_nms_data.append(temp)
        else:
            res_list_after_nms_data.append(temp)
    if len(res_list_nms) == 0:
        return result_list

    np_res_list_nms = np.array(res_list_nms)
    keep = py_cpu_nms(np_res_list_nms, inter_nms_thresh, iou_min_area, score_thresh)
    for tt in keep:
        res_list_after_nms_data.append(res_list_nms_data[tt])

    return res_list_after_nms_data

def compute_iou(rect1, rect2):
    '''
    函数说明：计算两个框的重叠面积
    输入：
    rec1 第一个框xmin ymin xmax ymax
    rec2 第二个框xmin ymin xmax ymax
    输出：
    iouv 重叠比例 0 没有
    '''
    xmin1, ymin1, xmax1, ymax1 = rect1
    xmin2, ymin2, xmax2, ymax2 = rect2
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    
    sum_area = s1 + s2

    left = max(xmin2, xmin1)
    right = min(xmax2, xmax1)
    top = max(ymin2, ymin1)
    bottom = min(ymax2, ymax1)

    if left >= right or top >= bottom:
        return 0,0

    intersection = (right - left) * (bottom - top)
    # print("-------------->intersection",intersection)
    #  iou in sum
    iou_normal =  intersection / (sum_area - intersection ) * 1.0
    # iou in bujian
    # print(s1, "-----", s2, "-----", intersection, "-----", intersection / s1 * 1.0)
    # return intersection / s1 * 1.0
    # iou in min
    small_area = min(s1, s2)
    iou_small =  intersection / small_area * 1.0
    return iou_normal, iou_small



def post_process(img, mmdet_result, bj_result, quexian_result, mmdet_labels, bj_labels, quexian_labels, post_process_label_list):
    """
    一张图所有缺陷类别合并nms输出
    对para.inter_class_nms_list_list中的类别，做类间nms
    Args:
        mmdet_result, bj_result, quexian_result：三个模型的输出结果
        mmdet_labels, bj_labels, quexian_labels: 三个模型的标签
        post_process_label_list：三个模型的标签集合（set无重复）
    Returns:
        output_result_list (list): 输出到文件的结果列表,格式如下[ (type,score,xmin,ymin,xmax,ymax), ]
   """
    discard_result_list = [mmdet_result, bj_result, quexian_result]
    discard_labels_list = [mmdet_labels, bj_labels, quexian_labels]
    # print('-----discard_result_list----', discard_result_list)
    for label in discard_detected_once_class_list:              
        for i in range(len(discard_labels_list)):
            src_single_result = find_single_class_result(
                discard_result_list[i], discard_labels_list[i], label)
            src_keep_list = []
            if (src_single_result is None) or (src_single_result.shape[0] <= 0):
                continue
            for j in range(3):
                if i == j:
                    continue
                dst_single_result = find_single_class_result(
                    discard_result_list[j], discard_labels_list[j], label)
                if (dst_single_result is None) or (dst_single_result.shape[0] <= 0):
                    continue
                for k in range(src_single_result.shape[0]):
                    src = src_single_result[k]
                    cur_max_iou = find_max_iou(
                        dst_single_result, src[0], src[1], src[2], src[3])
                    if cur_max_iou > discard_nms_thresh:
                        src_keep_list.append(k)

            if len(src_keep_list) > 0:
                set_single_class_result(
                    discard_result_list[i], discard_labels_list[i], label, src_single_result[src_keep_list])
            else:
                set_single_class_result(
                    discard_result_list[i], discard_labels_list[i], label, np.zeros((0, 5), dtype=np.float32))

    post_result_list = []
    label_result_merged_dict = {}
    for label in post_process_label_list:
        single_result_list = []
        mmdet_single_result = find_single_class_result(
            mmdet_result, mmdet_labels, label)
        if mmdet_single_result is not None:
            single_result_list.append(mmdet_single_result)
        bujian_single_result = find_single_class_result(
            bj_result, bj_labels, label)
        if bujian_single_result is not None:
            single_result_list.append(bujian_single_result)
        quexian_single_result = find_single_class_result(
            quexian_result, quexian_labels, label)
        if quexian_single_result is not None:
            single_result_list.append(quexian_single_result)

        merged_times = 0
        single_result_merged = np.zeros((0, 5), dtype=np.float32)
        for single_result in single_result_list:
            if (single_result is not None) and (single_result.shape[0] > 0):
                merged_times += 1
                single_result_merged = np.row_stack(
                    (single_result_merged, single_result))
        # if merged_times > 1:
        #     keep = py_cpu_nms(single_result_merged, nms_thresh, iou_min_area)
        #     single_result_merged = single_result_merged[keep]

        #############################same class nms######################################
        if label in nms_special_class_list:
            keep = py_cpu_nms(single_result_merged, nms_thresh_special_class, iou_min_area)
            single_result_merged = single_result_merged[keep]
        else:                                        
            keep = py_cpu_nms(single_result_merged, nms_thresh)
            single_result_merged = single_result_merged[keep]


        label_result_merged_dict[label] = single_result_merged


    # print('--------single_result_merged-----------', single_result_merged)

    for inter_class_nms_list in inter_class_nms_list_list:
        inter_class_nms(label_result_merged_dict, inter_class_nms_list)

    for inter_class_nms_list in special_inter_class_nms_list_list:
        inter_class_nms(label_result_merged_dict, inter_class_nms_list, score_thresh=0.2)

    # print('--------after inter nms-----------', single_result_merged)

    for label in defect_label_list:
        single_result_merged = label_result_merged_dict.get(                                                                                                                                                                                                                                                                                                                                                
            label, np.zeros((0, 5), dtype=np.float32))
        if single_result_merged.shape[0] > 0:
            for i in range(0, single_result_merged.shape[0]):
                # if label == "ylb":
                #     label = "SF6ylb"
                # print('----------single_result----------',single_result_merged[i])
                # if  (img.shape[1] >= single_result_merged[i][0] >= 0 and img.shape[0] >= single_result_merged[i][1] >= 0 and 
                #      img.shape[1] >= single_result_merged[i][2] >= 0 and img.shape[0] >= single_result_merged[i][3] >= 0):
                post_result_list.append([                                                                                                                                                                                                                                                                                               
                    label,
                    round(single_result_merged[i][4],2),
                    max(0,int(single_result_merged[i][0])),
                    max(0,int(single_result_merged[i][1])),
                    min(img.shape[1]-1, int(single_result_merged[i][2])),
                    min(img.shape[0]-1, int(single_result_merged[i][3]))])
    # print('----------post_result_list----------',post_result_list)
    sorted_result_list = sorted(
        post_result_list, key=lambda x: x[1], reverse=True)
    if max_output_num > 0:
        output_num = min(max_output_num, len(sorted_result_list))
        output_result_list = sorted_result_list[:output_num]
    else:
        output_result_list = sorted_result_list

    if ((img.shape[:2] == (1920, 1080)) and selection_of_1080p_filter) or ((img.shape[:2] == (1280, 720)) and selection_of_1080p_filter):
        output_result_list = limit_output_num_for_every_class(output_result_list, label_1080_num_limit_dict)
    else:
        output_result_list = limit_output_num_for_every_class(output_result_list, label_num_limit_dict)
    # print('------------output_result_list--------------', output_result_list)
    return output_result_list


def np_compute_iou(a_list, b_list):
    np_rects1 = np.array([row[2:] for row in a_list], dtype=np.float32) 
    np_rects2 = np.array([row[2:] for row in b_list], dtype=np.float32) 
    np_area1 = (np_rects1[:, 2] - np_rects1[:, 0] + 1)*(np_rects1[:, 3] - np_rects1[:, 1] + 1)
    np_area2 = (np_rects2[:, 2] - np_rects2[:, 0] + 1)*(np_rects2[:, 3] - np_rects2[:, 1] + 1)
    xx1 = np.maximum(np_rects1[:, None, 0], np_rects2[:, 0])
    yy1 = np.maximum(np_rects1[:, None, 1], np_rects2[:, 1])
    xx2 = np.minimum(np_rects1[:, None, 2], np_rects2[:, 2])
    yy2 = np.minimum(np_rects1[:, None, 3], np_rects2[:, 3])
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h   
    union = np_area1[:, None] + np_area2[:] - inter
    np_min_area = np.minimum(np_area1[:, None], np_area2[:]) 
    eps = np.finfo(np.float32).eps
    ovr_union = inter / np.maximum(union, eps)
    ovr_min_area = inter/ np.maximum(np_min_area, eps)
    return ovr_union, ovr_min_area


def inference_detector_with_conf_thresh(model, img, label_conf_thresh_dict, classes):
    """
    根据置信度阈值过滤模型推理结果

    Args:
        img (numpy): 原图
        model：模型
        label_conf_thresh_dict: label和conf_thresh组成的字典,
        可以为None,也可以只设置部分label的阈值；没有设置阈值的label,输出全部结果；

    Returns:
        result(list(numpy((N,5),dtype=np.float32))): 结果列表,可能为空列表
    """
    result = inference_detector(model, img)
    if label_conf_thresh_dict is None:
        return result
    for idx, res in enumerate(result):
        label = classes[idx]
        if label in label_conf_thresh_dict:
            conf_thresh = label_conf_thresh_dict.get(label, 0.00)
            result[idx] = res[np.where(res[:, 4] >= conf_thresh)]

    return result


def bujian_merge_quexian(output_result_list, bujian_classes, qx_classes, quexian_classes):
    bujian_output_result = []
    quexian_output_result = []
    qx_output_result = []
    
    for res in output_result_list:
        if res[0] in bujian_classes:
            bujian_output_result.append(res)
            continue
        if res[0] not in bujian_classes:
            qx_output_result.append(res)
        if res[0] in quexian_classes:
            quexian_output_result.append(res)
    if len(bujian_output_result) > 0 and len(quexian_output_result) > 0:       
        ovr_union, ovr_min_area = np_compute_iou(bujian_output_result, quexian_output_result)  
        inds_union = np.where(ovr_union>bujian_iou_normal)
        inds_min_area = np.where(ovr_min_area>bujian_iou_small)
        for bujian_index in range(len(bujian_output_result)):
            np_union_index = np.where(inds_union[0] == bujian_index)[0]
            np_min_area_index = np.where(inds_min_area[0] == bujian_index)[0]
            quexian_index = np.inf
            if np_union_index.size > 0:
                tmp_index = inds_union[1][np_union_index[0]]
                quexian_index = min(tmp_index,quexian_index)
            if np_min_area_index.size >0:
                tmp_index = inds_min_area[1][np_min_area_index[0]]
                quexian_index = min(tmp_index,quexian_index)
            if quexian_index<np.inf:
                bujian = bujian_output_result[bujian_index]
                quexian = quexian_output_result[quexian_index]
                bujian[1] = round((quexian[1] + bujian[1]) / 2, 2)
                qx_output_result.append(bujian)
        output_result_list = qx_output_result
    elif len(bujian_output_result) > 0 and len(quexian_output_result) == 0:
        output_result_list = qx_output_result
    else:
        output_result_list = output_result_list
    return output_result_list


def bujian_merge_quexian_nonp(output_result_list, bujian_classes, qx_classes, quexian_classes):
    bujian_output_result = []
    quexian_output_result = []
    qx_output_result = []
    
    for res in output_result_list:
        if res[0] in bujian_classes:
            bujian_output_result.append(res)
            continue
        if res[0] not in bujian_classes:
            qx_output_result.append(res)
        if res[0] in quexian_classes:
            quexian_output_result.append(res)
    if len(bujian_output_result) > 0 and len(quexian_output_result) > 0:       
        for bujian in bujian_output_result:
            rect1 = bujian[2:]
            for quexian in quexian_output_result:
                rect2 = quexian[2:]
                iou_noraml, iou_small = compute_iou(rect1, rect2)
                if iou_noraml >bujian_iou_normal or iou_small > bujian_iou_small:
                    bujian[1] = round((quexian[1] + bujian[1]) / 2, 2)
                    qx_output_result.append(bujian)
                    break
        output_result_list = qx_output_result
    elif len(bujian_output_result) > 0 and len(quexian_output_result) == 0:
        output_result_list = qx_output_result
    else:
        output_result_list = output_result_list
    return output_result_list


def bujian_merge_quexian_with_sly(output_result_list, bujian_classes, qx_classes, quexian_classes, img):
    bujian_output_result = []
    quexian_output_result = []
    qx_output_result = []
    
    for res in output_result_list:
        if res[0] in bujian_classes:
            bujian_output_result.append(res)
            continue
        if res[0] not in bujian_classes:
            qx_output_result.append(res)
        if res[0] in quexian_classes:
            quexian_output_result.append(res)
    if len(bujian_output_result) > 0 and len(quexian_output_result) > 0: 
        q_result = []
        for quexian in  quexian_output_result:
            width =  quexian[4] - quexian[2]
            height = quexian[5] - quexian[3]
            rate = 0.1
            quexian = [quexian[0], quexian[1], max(0, quexian[2]-rate*width), max(0,quexian[3]-rate*height), min(img.shape[1]-1,quexian[4]+rate*width), min(img.shape[0]-1,quexian[5]+rate*height)]
            q_result.append(quexian)
        quexian_output_result = q_result
        # print(quexian_output_result,66666666)
        ovr_union, ovr_min_area = np_compute_iou(bujian_output_result, quexian_output_result)  
        inds_union = np.where(ovr_union>bujian_iou_normal)
        inds_min_area = np.where(ovr_min_area>bujian_iou_small)
        for bujian_index in range(len(bujian_output_result)):
            np_union_index = np.where(inds_union[0] == bujian_index)[0]
            np_min_area_index = np.where(inds_min_area[0] == bujian_index)[0]
            quexian_index = np.inf
            if np_union_index.size > 0:
                tmp_index = inds_union[1][np_union_index[0]]
                quexian_index = min(tmp_index,quexian_index)
            if np_min_area_index.size >0:
                tmp_index = inds_min_area[1][np_min_area_index[0]]
                quexian_index = min(tmp_index,quexian_index)
            if quexian_index<np.inf:
                bujian = bujian_output_result[bujian_index]
                quexian = quexian_output_result[quexian_index]
                bujian[1] = round((quexian[1] + bujian[1]) / 2, 2)
                qx_output_result.append(bujian)
        output_result_list = qx_output_result
    elif len(bujian_output_result) > 0 and len(quexian_output_result) == 0:
        output_result_list = qx_output_result
    else:
        output_result_list = output_result_list
    return output_result_list


def bujian_merge_quexian_with_single_quexian(output_result_list, bujian_classes, qx_classes, quexian_classes, bujian_copy_dict):
    bujian_output_result = []
    quexian_output_result = []
    qx_output_result = []
    
    for res in output_result_list:
        if res[0] in bujian_classes:
            bujian_output_result.append(res)
            continue
        if res[0] not in bujian_classes:
            qx_output_result.append(res)
        if res[0] in quexian_classes:
            quexian_output_result.append(res)
    if len(bujian_output_result) > 0 and len(quexian_output_result) > 0:       
        for bujian in bujian_output_result:
            rect1 = bujian[2:]
            for quexian in quexian_output_result:
                rect2 = quexian[2:]
                iou_noraml, iou_small = compute_iou(rect1, rect2)
                if iou_noraml >bujian_iou_normal or iou_small > bujian_iou_small:
                    bujian[1] = round((quexian[1] + bujian[1]) / 2, 2)
                    qx_output_result.append(bujian)
                    break
        output_result_list = qx_output_result
    elif len(bujian_output_result) > 0 and len(quexian_output_result) == 0:
        output_result_list = qx_output_result
    elif len(bujian_output_result)  == 0 and len(quexian_output_result) > 0:
        # bujian_copy_dict = {'bjdsyc_ywj':'ywj', 'bjdsyc_ywc':'ywc', 'ws_ywyc':'cysb_qtjdq'}
        for quexian in output_result_list:
            if quexian[0] in bujian_copy_dict:
                bujian = [bujian_copy_dict.get(quexian[0]), quexian[1], quexian[2], quexian[3], quexian[4], quexian[5]]
                output_result_list.append(bujian)
        output_result_list = output_result_list
    else:
        output_result_list = output_result_list
    return output_result_list


def cord_transfer(result_list, img, input_h, input_w,):
    model_h, model_w = preprocess_image(input_w, input_h, img)
    ratio_h = model_h / img.shape[0];
    ratio_w = model_w / img.shape[1];

    for item in result_list:
        # box = item[2:]
        # box = [int(box[0] / ratio_w), int(box[1] / ratio_h), int(box[2] / ratio_w), int(box[3] / ratio_h)]
        # box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        # item[2:] = box
        item[4] = item[4] - item[2]
        item[5] = item[5] - item[3]
        if (ratio_h > ratio_w):
            item[2] = item[2] / ratio_w;
            item[3] = (item[3] - (model_h - ratio_w * img.shape[0]) / 2) / ratio_w;
            item[4] = item[4] / ratio_w;
            item[5] = item[5] / ratio_w;
        else:
            item[2] = (item[2] - (model_w - ratio_h * img.shape[1]) / 2) / ratio_h;
            item[3] = item[3] / ratio_h;
            item[4] = item[4] / ratio_h;
            item[5] = item[5] / ratio_h;
        item[2:] = [item[2], item[3], item[2]+item[4],item[3]+item[5]]
    return result_list




def preprocess_image(input_w, input_h, raw_bgr_image):
    """
    description: Convert BGR image to RGB,
                    resize and pad it to target size, normalize to [0,1],
                    transform to NCHW format.
    param:
        input_image_path: str, image path
    return:
        image:  the processed image
        image_raw: the original image
        h: original height
        w: original width
    """
    image_raw = raw_bgr_image
    h, w, c = image_raw.shape
    # print('-------------init scale-------------:{}\n'.format((h,w)))
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = input_w / w
    r_h = input_h / h
    if r_h > r_w:
        tw = input_w
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((input_h - th) / 2)
        ty2 = input_h - th - ty1
    else:
        tw = int(r_h * w)
        th = input_h
        tx1 = int((input_w - tw) / 2)
        tx2 = input_w - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # print(tw,th)
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    # print('-------------final scale-------------:{}\n'.format((image.shape)))
    model_h, model_w = image.shape[3],image.shape[2]
    return model_h, model_w