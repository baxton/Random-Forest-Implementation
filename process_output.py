


import re



def process():
    fn = "c:\\Temp\\kaggle\\epilepsy\\scripts\\sub6\\2.txt"

    sensors = {}
    tmp = {}

    with open(fn, "r") as fin:
        cnt_lines = 0
        line = fin.readline()
        while line:
            cnt_lines += 1
            #
            line = line.strip()

            if "sensor" in line:
                k = re.findall("# sensor  [0-9][0-9]?", line)[0]
                #print line

                v = float(re.findall("#.*\\[ (.*).*\\]", line)[0])
                if k in tmp:
                    item = tmp[k]
                    tmp[k] = (v + item[0], item[1] + 1)
                else:
                    tmp[k] = (v, 1)

            elif "Dog" in line:
                if 'preictal' in line:
                    for k, v in tmp.items():
                        kex = k + " P"
                        if kex in sensors:
                            tmp_item = tmp[k]
                            item = sensors[kex]
                            sensors[kex] = (item[0] + tmp_item[0], item[1] + tmp_item[1])
                        else:
                            tmp_item = tmp[k]
                            sensors[kex] = tmp_item
                else:
                    for k, v in tmp.items():
                        kex = k + " I"
                        if kex in sensors:
                            tmp_item = tmp[k]
                            item = sensors[kex]

                            num_zeros = tmp_item[1] - tmp_item[0]
                            sensors[kex] = (item[0] + num_zeros, item[1] + tmp_item[1])
                        else:
                            tmp_item = tmp[k]
                            num_zeros = tmp_item[1] - tmp_item[0]
                            sensors[kex] = (num_zeros, tmp_item[1])
                tmp.clear()
            else:
                tmp.clear()
            #
            line = fin.readline()


    print cnt_lines

    mn = max([float(v)/n for v, n in sensors.values()])

    P = mn / 100.
    keys = sensors.keys()
    keys.sort()
    for k in keys:
        v = sensors[k]
        recall = float(v[0]) / v[1]

        N = recall * P

        print "%s %0.4f\t" % (k, recall), "|" * int(N)

    print "=========="

    for i in range(0, len(keys), 2):
        item1 = sensors[ keys[i+0] ]
        item2 = sensors[ keys[i+1] ]

        v1 = float(item1[0]) / item1[1]
        v2 = float(item2[0]) / item2[1]

        f1 = 2 * (v1 * v2) / (v1 + v2)

        print "%s (%s) %0.4f\t" % (keys[i], keys[i+1], f1)





def main():
    process()

if __name__ == '__main__':
    main()
