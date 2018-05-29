import csv

upd_dict = {}

with open ('20180523_msg_id_scene.txt','r') as update:
    reader = csv.reader(update)
    next(reader)
    for each in reader:
        upd=each[0].split('\t')
        upd_dict[upd[0]] = upd[1]

list_of_sms =[]

with open ('sms_messages.csv','r') as sms_m:
    reader = csv.reader(sms_m)
    # writer = csv.writer(sms_m)
    next(reader)
    for each in reader:
        list_of_sms.append(each)
    for each in list_of_sms:
        if each[0] in upd_dict:
            each[2] = upd_dict[each[0]]

with open ('sms_messages.csv','w') as sms_m:
    writer = csv.writer(sms_m)
    writer.writerow(['id','raw_content','scene_code'])
    for each in list_of_sms:
        writer.writerow(each)

