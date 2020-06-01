import sys
import os
import gc
from datetime import datetime
import cv2 as cv

from imageRegistration import image_registration as ir


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', content=""):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    if iteration == total:
        sys.stdout.write('\r%s |%s| %s%% %s %s' % (prefix, bar, percent, content, suffix))
        sys.stdout.flush()
        print()
    else:
        sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, content))
        sys.stdout.flush()


def printResult(set_test, set_pass, set_fail):
    print("Count test set:", len(set_test))
    print("Count test pass:", len(set_pass))
    print("\t", set_pass)
    print("Count test fail:", len(set_fail))
    print("\t", set_fail)
    print("Accuracy: ", round(len(set_pass) / (len(set_pass) + len(set_fail)) * 100, 2), "%", sep="")


def writeResultImg(is_pass, name, root, image, algorithm, time):
    if not os.path.isdir(root):
        os.makedirs(root, exist_ok=True)
    f = open(root+"/"+root+".txt", "a")
    if is_pass:
        text = "pass"
    else:
        text = "fail"
    space = "\t"
    f.write(name + space + text + space + algorithm + space + str(time) + "\n")
    cv.imwrite(root + "/" + text + "-" + name + ".jpg", image)


root_test_path = "20190409_11-926/"

list_set_test = []
for x in os.listdir(root_test_path):
    set_test = os.listdir(root_test_path + x)
    if len(set_test) == 2:
        list_set_test.append([x, set_test[0], set_test[1]])
# tp = tn = fp = fn = 0

print("== NORMAL TEST ================================================================================================")
progress_total = len(list_set_test)
now = datetime.now()
prefix_folder_name = now.strftime("%d%m%Y%H%M%S")

print("== Using SIFT Algorithm ==")
tn = tp = 0
folder_name = prefix_folder_name + "_result_sift"
count = 0
set_test_pass = []
set_test_fail = []
printProgressBar(count, progress_total,
                 prefix='Progress:',
                 suffix='Complete',
                 length=50,
                 content="(Pass: " + str(tp) + " fail: " + str(tn) + " || total: " + str(progress_total) + ")")
for set_test in list_set_test:
    start = datetime.now()
    predict, img = ir(root_test_path + set_test[0] + "/" + set_test[1],
                      root_test_path + set_test[0] + "/" + set_test[2])
    end = datetime.now()
    if predict:
        tp += 1
        set_test_pass.append(set_test[0])
    else:
        tn += 1
        set_test_fail.append(set_test[0])
    writeResultImg(predict, set_test[0], folder_name, img, "SIFT", end-start)
    count += 1
    printProgressBar(count, progress_total,
                     prefix='Progress:',
                     suffix='Complete',
                     length=50,
                     content="(Pass: " + str(tp) + " fail: " + str(tn) + " || total: " + str(progress_total) + ")")
printResult(list_set_test, set_test_pass, set_test_fail)
print()
gc.collect()

print("== Using SURF Algorithm ==")
tn = tp = 0
folder_name = prefix_folder_name + "_result_surf"
count = 0
set_test_pass = []
set_test_fail = []
printProgressBar(count, progress_total,
                 prefix='Progress:',
                 suffix='Complete',
                 length=50,
                 content="(Pass: " + str(tp) + " fail: " + str(tn) + " || total: " + str(progress_total) + ")")
for set_test in list_set_test:
    start = datetime.now()
    predict, img = ir(root_test_path + set_test[0] + "/" + set_test[1],
                      root_test_path + set_test[0] + "/" + set_test[2],
                      algorithm=2)
    end = datetime.now()
    if predict:
        tp += 1
        set_test_pass.append(set_test[0])
    else:
        tn += 1
        set_test_fail.append(set_test[0])
    writeResultImg(predict, set_test[0], folder_name, img, "SURF", end-start)
    count += 1
    printProgressBar(count, progress_total,
                     prefix='Progress:',
                     suffix='Complete',
                     length=50,
                     content="(Pass: " + str(tp) + " fail: " + str(tn) + " || total: " + str(progress_total) + ")")
printResult(list_set_test, set_test_pass, set_test_fail)
print()
gc.collect()

print("== Using BRISK Algorithm ==")
tn = tp = 0
folder_name = prefix_folder_name + "_result_brisk"
count = 0
set_test_pass = []
set_test_fail = []
printProgressBar(count, progress_total,
                 prefix='Progress:',
                 suffix='Complete',
                 length=50,
                 content="(Pass: " + str(tp) + " fail: " + str(tn) + " || total: " + str(progress_total) + ")")
for set_test in list_set_test:
    start = datetime.now()
    predict, img = ir(root_test_path + set_test[0] + "/" + set_test[1],
                      root_test_path + set_test[0] + "/" + set_test[2],
                      algorithm=3)
    end = datetime.now()
    if predict:
        tp += 1
        set_test_pass.append(set_test[0])
    else:
        tn += 1
        set_test_fail.append(set_test[0])
    writeResultImg(predict, set_test[0], folder_name, img, "BRISK", end-start)
    count += 1
    printProgressBar(count, progress_total,
                     prefix='Progress:',
                     suffix='Complete',
                     length=50,
                     content="(Pass: " + str(tp) + " fail: " + str(tn) + " || total: " + str(progress_total) + ")")
printResult(list_set_test, set_test_pass, set_test_fail)
print()
gc.collect()

print("== Using BRISK Algorithm with scale option ==")
tn = tp = 0
folder_name = prefix_folder_name + "_result_brisk_improve"
count = 0
set_test_pass = []
set_test_fail = []
printProgressBar(count, progress_total,
                 prefix='Progress:',
                 suffix='Complete',
                 length=50,
                 content="(Pass: " + str(tp) + " fail: " + str(tn) + " || total: " + str(progress_total) + ")")
for set_test in list_set_test:
    start = datetime.now()
    predict, img = ir(root_test_path + set_test[0] + "/" + set_test[1],
                      root_test_path + set_test[0] + "/" + set_test[2],
                      algorithm=4)
    end = datetime.now()
    if predict:
        tp += 1
        set_test_pass.append(set_test[0])
    else:
        tn += 1
        set_test_fail.append(set_test[0])
    writeResultImg(predict, set_test[0], folder_name, img, "BRISK_improve", end-start)
    count += 1
    printProgressBar(count, progress_total,
                     prefix='Progress:',
                     suffix='Complete',
                     length=50,
                     content="(Pass: " + str(tp) + " fail: " + str(tn) + " || total: " + str(progress_total) + ")")
printResult(list_set_test, set_test_pass, set_test_fail)
print()
gc.collect()

print("== Using all Algorithm ==")
tn = tp = 0
folder_name = prefix_folder_name + "_result_all"
count = 0
set_test_pass = []
set_test_fail = []
printProgressBar(count, progress_total,
                 prefix='Progress:',
                 suffix='Complete',
                 length=50,
                 content="(Pass: " + str(tp) + " fail: " + str(tn) + " || total: " + str(progress_total) + ")")
for set_test in list_set_test:
    start = datetime.now()
    predict, img = ir(root_test_path + set_test[0] + "/" + set_test[1],
                      root_test_path + set_test[0] + "/" + set_test[2],
                      algorithm=1)
    gc.collect()
    if not predict:
        predict, img = ir(root_test_path + set_test[0] + "/" + set_test[1],
                          root_test_path + set_test[0] + "/" + set_test[2],
                          algorithm=2)
    gc.collect()
    if not predict:
        predict, img = ir(root_test_path + set_test[0] + "/" + set_test[1],
                          root_test_path + set_test[0] + "/" + set_test[2],
                          algorithm=3)
    gc.collect()
    if not predict:
        predict, img = ir(root_test_path + set_test[0] + "/" + set_test[1],
                          root_test_path + set_test[0] + "/" + set_test[2],
                          algorithm=4)
    gc.collect()
    end = datetime.now()
    if predict:
        tp += 1
        set_test_pass.append(set_test[0])
    else:
        tn += 1
        set_test_fail.append(set_test[0])
    writeResultImg(predict, set_test[0], folder_name, img, "all", end - start)
    count += 1
    printProgressBar(count, progress_total,
                     prefix='Progress:',
                     suffix='Complete',
                     length=50,
                     content="(Pass: " + str(tp) + " fail: " + str(tn) + " || total: " + str(progress_total) + ")")
printResult(list_set_test, set_test_pass, set_test_fail)
print()
gc.collect()

# print("== CROSS TEST =================================================================================================")
# progress_total = 0
# for i in range(len(list_set_test)):
#     progress_total += i
# progress_total *= 4
# count = 0
# printProgressBar(count, progress_total,
#                  prefix='Progress:',
#                  suffix='Complete',
#                  length=50,
#                  content="(Pass: " + str(fn) + " fail: " + str(fp) + " || total: " + str(progress_total) + ")")
# cross_test_pass = []
# cross_test_fail = []
# for i in range(len(list_set_test)):
#     for j in range(i + 1, len(list_set_test)):
#         identity, img = ir(root_test_path + list_set_test[i][0] + "/" + list_set_test[i][1],
#                            root_test_path + list_set_test[j][0] + "/" + list_set_test[j][1])
#         if identity:
#             fp += 1
#             cross_test_fail.append([list_set_test[i][0] + "/" + list_set_test[i][1],
#                                     list_set_test[j][0] + "/" + list_set_test[j][1]])
#             cv.imwrite("result_cross/" +
#                        list_set_test[i][0] + "_" + list_set_test[i][1] +
#                        "To" +
#                        list_set_test[j][0] + "_" + list_set_test[j][1] +
#                        "-fail.jpg", img)
#         else:
#             cross_test_pass.append([list_set_test[i][0] + "/" + list_set_test[i][1],
#                                     list_set_test[j][0] + "/" + list_set_test[j][1]])
#             fn += 1
#             cv.imwrite("result_cross/" +
#                        list_set_test[i][0] + "_" + list_set_test[i][1] +
#                        "To" +
#                        list_set_test[j][0] + "_" + list_set_test[j][1] +
#                        "-pass.jpg", img)
#         count += 1
#         printProgressBar(count, progress_total,
#                          prefix='Progress:',
#                          suffix='Complete',
#                          length=50,
#                          content="(Pass: " + str(fn) + " fail: " + str(fp) + " || total: " + str(progress_total) + ")")
#         identity, img = ir(root_test_path + list_set_test[i][0] + "/" + list_set_test[i][1],
#                            root_test_path + list_set_test[j][0] + "/" + list_set_test[j][2])
#         if identity:
#             fp += 1
#             cross_test_fail.append([list_set_test[i][0] + "/" + list_set_test[i][1],
#                                     list_set_test[j][0] + "/" + list_set_test[j][2]])
#             cv.imwrite("result_cross/" +
#                        list_set_test[i][0] + "_" + list_set_test[i][1] +
#                        "To" +
#                        list_set_test[j][0] + "_" + list_set_test[j][2] +
#                        "-fail.jpg", img)
#         else:
#             cross_test_pass.append([list_set_test[i][0] + "/" + list_set_test[i][1],
#                                     list_set_test[j][0] + "/" + list_set_test[j][2]])
#             fn += 1
#             cv.imwrite("result_cross/" +
#                        list_set_test[i][0] + "_" + list_set_test[i][1] +
#                        "To" +
#                        list_set_test[j][0] + "_" + list_set_test[j][2] +
#                        "-pass.jpg", img)
#         count += 1
#         printProgressBar(count, progress_total,
#                          prefix='Progress:',
#                          suffix='Complete',
#                          length=50,
#                          content="(Pass: " + str(fn) + " fail: " + str(fp) + " || total: " + str(progress_total) + ")")
#         identity, img = ir(root_test_path + list_set_test[i][0] + "/" + list_set_test[i][2],
#                            root_test_path + list_set_test[j][0] + "/" + list_set_test[j][1])
#         if identity:
#             fp += 1
#             cross_test_fail.append([list_set_test[i][0] + "/" + list_set_test[i][2],
#                                     list_set_test[j][0] + "/" + list_set_test[j][1]])
#             cv.imwrite("result_cross/" +
#                        list_set_test[i][0] + "_" + list_set_test[i][2] +
#                        "To" +
#                        list_set_test[j][0] + "_" + list_set_test[j][1] +
#                        "-fail.jpg", img)
#         else:
#             cross_test_pass.append([list_set_test[i][0] + "/" + list_set_test[i][2],
#                                     list_set_test[j][0] + "/" + list_set_test[j][1]])
#             fn += 1
#             cv.imwrite("result_cross/" +
#                        list_set_test[i][0] + "_" + list_set_test[i][2] +
#                        "To" +
#                        list_set_test[j][0] + "_" + list_set_test[j][1] +
#                        "-pass.jpg", img)
#         count += 1
#         printProgressBar(count, progress_total,
#                          prefix='Progress:',
#                          suffix='Complete',
#                          length=50,
#                          content="(Pass: " + str(fn) + " fail: " + str(fp) + " || total: " + str(progress_total) + ")")
#         identity, img = ir(root_test_path + list_set_test[i][0] + "/" + list_set_test[i][2],
#                            root_test_path + list_set_test[j][0] + "/" + list_set_test[j][2])
#         if identity:
#             fp += 1
#             cross_test_fail.append([list_set_test[i][0] + "/" + list_set_test[i][2],
#                                     list_set_test[j][0] + "/" + list_set_test[j][2]])
#             cv.imwrite("result_cross/" +
#                        list_set_test[i][0] + "_" + list_set_test[i][2] +
#                        "To" +
#                        list_set_test[j][0] + "_" + list_set_test[j][2] +
#                        "-fail.jpg", img)
#         else:
#             cross_test_pass.append([list_set_test[i][0] + "/" + list_set_test[i][2],
#                                     list_set_test[j][0] + "/" + list_set_test[j][2]])
#             fn += 1
#             cv.imwrite("result_cross/" +
#                        list_set_test[i][0] + "_" + list_set_test[i][2] +
#                        "To" +
#                        list_set_test[j][0] + "_" + list_set_test[j][2] +
#                        "-pass.jpg", img)
#         count += 1
#         printProgressBar(count, progress_total,
#                          prefix='Progress:',
#                          suffix='Complete',
#                          length=50,
#                          content="(Pass: " + str(fn) + " fail: " + str(fp) + " || total: " + str(progress_total) + ")")
#
# print("Count test set:", fp + fn)
# print("Count test pass:", fn)
# print("Count test fail:", fp)
# print("Accuracy: ", round(fn / (fn + fp) * 100, 2), "%", sep="")
# print("Pass:")
# print(cross_test_pass)
# print("Fail:")
# print(cross_test_fail)
