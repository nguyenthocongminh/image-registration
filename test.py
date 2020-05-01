import os
import cv2 as cv
import sys
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


root_test_path = "rrr/"
list_set_test = []
for x in os.listdir(root_test_path):
    set_test = os.listdir(root_test_path + x)
    list_set_test.append([x, set_test[0], set_test[1]])
tp = tn = fp = fn = 0

print("NORMAL")
progress_total = len(list_set_test)
count = 0
printProgressBar(count, progress_total,
                 prefix='Progress:',
                 suffix='Complete',
                 length=50,
                 content="(Pass: " + str(tp) + " fail: " + str(tn) + " || total: " + str(progress_total) + ")")
set_test_pass = []
set_test_fail = []
for set_test in list_set_test:
    identity, img = ir(root_test_path + set_test[0] + "/" + set_test[1],
                       root_test_path + set_test[0] + "/" + set_test[2])
    if identity:
        tp += 1
        set_test_pass.append(set_test[0])
        cv.imwrite("result/" + set_test[0] + "-pass.jpg", img)
    else:
        tn += 1
        set_test_fail.append(set_test[0])
        cv.imwrite("result/" + set_test[0] + "-fail.jpg", img)
    count += 1
    printProgressBar(count, progress_total,
                     prefix='Progress:',
                     suffix='Complete',
                     length=50,
                     content="(Pass: " + str(tp) + " fail: " + str(tn) + " || total: " + str(progress_total) + ")")

print("CROSS")
progress_total = 0
for i in range(len(list_set_test)):
    progress_total += i
progress_total *= 4
count = 0
printProgressBar(count, progress_total,
                 prefix='Progress:',
                 suffix='Complete',
                 length=50,
                 content="(Pass: " + str(fn) + " fail: " + str(fp) + " || total: " + str(progress_total) + ")")
cross_test_pass = []
cross_test_fail = []
for i in range(len(list_set_test)):
    for j in range(i + 1, len(list_set_test)):
        identity, img = ir(root_test_path + list_set_test[i][0] + "/" + list_set_test[i][1],
                           root_test_path + list_set_test[j][0] + "/" + list_set_test[j][1])
        if identity:
            fp += 1
            cross_test_fail.append([list_set_test[i][0] + "/" + list_set_test[i][1],
                                    list_set_test[j][0] + "/" + list_set_test[j][1]])
        else:
            cross_test_pass.append([list_set_test[i][0] + "/" + list_set_test[i][1],
                                    list_set_test[j][0] + "/" + list_set_test[j][1]])
            fn += 1
        count += 1
        printProgressBar(count, progress_total,
                         prefix='Progress:',
                         suffix='Complete',
                         length=50,
                         content="(Pass: " + str(fn) + " fail: " + str(fp) + " || total: " + str(progress_total) + ")")
        identity, img = ir(root_test_path + list_set_test[i][0] + "/" + list_set_test[i][1],
                           root_test_path + list_set_test[j][0] + "/" + list_set_test[j][2])
        if identity:
            fp += 1
            cross_test_fail.append([list_set_test[i][0] + "/" + list_set_test[i][1],
                                    list_set_test[j][0] + "/" + list_set_test[j][2]])
        else:
            cross_test_pass.append([list_set_test[i][0] + "/" + list_set_test[i][1],
                                    list_set_test[j][0] + "/" + list_set_test[j][2]])
            fn += 1
        count += 1
        printProgressBar(count, progress_total,
                         prefix='Progress:',
                         suffix='Complete',
                         length=50,
                         content="(Pass: " + str(fn) + " fail: " + str(fp) + " || total: " + str(progress_total) + ")")
        identity, img = ir(root_test_path + list_set_test[i][0] + "/" + list_set_test[i][2],
                           root_test_path + list_set_test[j][0] + "/" + list_set_test[j][1])
        if identity:
            fp += 1
            cross_test_fail.append([list_set_test[i][0] + "/" + list_set_test[i][2],
                                    list_set_test[j][0] + "/" + list_set_test[j][1]])
        else:
            cross_test_pass.append([list_set_test[i][0] + "/" + list_set_test[i][2],
                                    list_set_test[j][0] + "/" + list_set_test[j][1]])
            fn += 1
        count += 1
        printProgressBar(count, progress_total,
                         prefix='Progress:',
                         suffix='Complete',
                         length=50,
                         content="(Pass: " + str(fn) + " fail: " + str(fp) + " || total: " + str(progress_total) + ")")
        identity, img = ir(root_test_path + list_set_test[i][0] + "/" + list_set_test[i][2],
                           root_test_path + list_set_test[j][0] + "/" + list_set_test[j][2])
        if identity:
            fp += 1
            cross_test_fail.append([list_set_test[i][0] + "/" + list_set_test[i][2],
                                    list_set_test[j][0] + "/" + list_set_test[j][2]])
        else:
            cross_test_pass.append([list_set_test[i][0] + "/" + list_set_test[i][2],
                                    list_set_test[j][0] + "/" + list_set_test[j][2]])
            fn += 1
        count += 1
        printProgressBar(count, progress_total,
                         prefix='Progress:',
                         suffix='Complete',
                         length=50,
                         content="(Pass: " + str(fn) + " fail: " + str(fp) + " || total: " + str(progress_total) + ")")

print()
print("== NORMAL TEST ================================================================================================")
print("Count test set:", len(list_set_test))
print("Count test pass:", len(set_test_pass))
print("\t", set_test_pass)
print("Count test fail:", len(set_test_fail))
print("\t", set_test_fail)
print("Accuracy: ", round(len(set_test_pass) / (len(set_test_pass) + len(set_test_fail)) * 100, 2), "%", sep="")
print()
print("== CROSS TEST =================================================================================================")
print("Count test set:", fp + fn)
print("Count test pass:", fn)
print("Count test fail:", fp)
print("Accuracy: ", round(fn / (fn + fp) * 100, 2), "%", sep="")
print("Pass:")
print(cross_test_pass)
print("Fail:")
print(cross_test_fail)
