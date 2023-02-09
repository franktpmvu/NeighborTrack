import cv2
import numpy as np
import json


#dir_path = '/data/TracKit/dataset/VOT2018/basketball/'
#dir_path = '/data/TracKit/dataset/VOT2018/dinosaur/'
#dir_path = '/data/TracKit/dataset/LaSOT/chameleon/chameleon-6/img/'

#dir_path = '/data/TracKit/dataset/LaSOT/cup/cup-1/img/'
#dir_path = '/data/TracKit/dataset/LaSOT/lion/lion-5/img/'

# choose codec according to format needed

def convert_to_video(img_dir,output_name,img_list):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    img = cv2.imread(img_dir+img_list[0])
    print(img_dir+img_list[0])
    video = cv2.VideoWriter(output_name, fourcc, 10.0, (img.shape[1], img.shape[0]))

    for j in img_list:
        img = cv2.imread(dir_path+j)
        if img is None:
            print('file {} not found'.format(dir_path+j))
            break
        video.write(img)

    cv2.destroyAllWindows()
    video.release()






def create_dataset_dict(base_path):
    all_datasets=[]

    f=open(base_path+'/train/_annotations.coco.json')
    data=json.load(f)
    all_datasets.append(data)
    f=open(base_path+'/valid/_annotations.coco.json')
    data=json.load(f)
    all_datasets.append(data)
    f=open(base_path+'/test/_annotations.coco.json')
    data=json.load(f)
    all_datasets.append(data)



    filename_list=[]
    videoId_list=[]
    frameId_list=[]
    imageId_list=[]
    all_xywh_id=None
    start_id=0

    for data in all_datasets:

        for idx,_data in enumerate(data['images']):
            if not _data['id']==idx:
                print(_data['image_id'])
                print(idx)
                break
            _,_,video_id,frame_id,_=_data['file_name'].split('_')

            filename_list.append(_data['file_name'])
            videoId_list.append(int(video_id))
            frameId_list.append(int(frame_id))
            imageId_list.append(int(_data['id']))

        xywh_id=np.zeros([len(data['images']),4])-1

        for idx,_data in enumerate(data['annotations']):
            if not _data['image_id']==imageId_list[start_id+_data['image_id']]:
                print(_data['image_id'])
                print(_data['image_id'])
                print(dsadsada)
                break
            xywh_id[_data['image_id'],:]=_data['bbox']

        if all_xywh_id is None:
            all_xywh_id = xywh_id
        else:
            all_xywh_id = np.concatenate((all_xywh_id,xywh_id),axis=0)
        start_id+=len(data['images'])








    d_videos=dict()

    for idx,videoId in enumerate(videoId_list):
        if not str(videoId) in d_videos:
            d_videos[str(videoId)]=dict()
            d_videos[str(videoId)]['frameId']=[]
            d_videos[str(videoId)]['xywh']=[]
            d_videos[str(videoId)]['filename']=[]
        else:
            d_videos[str(videoId)]['frameId'].append(frameId_list[idx])
            d_videos[str(videoId)]['xywh'].append(all_xywh_id[idx,:])
            d_videos[str(videoId)]['filename'].append(filename_list[idx])


    #print(d_videos.keys())



    for _key in d_videos.keys():
        frameid=[]
        xywh=[]
        filename=[]
        sorted_index = sorted(range(len(d_videos[_key]['frameId'])), key = lambda k : d_videos[_key]['frameId'][k]) 
        #print(sorted_index)

        for index in sorted_index :
            frameid.append(d_videos[_key]['frameId'][index])
            xywh.append(d_videos[_key]['xywh'][index])
            filename.append(d_videos[_key]['filename'][index])
        d_videos[_key]['frameId']=frameid
        d_videos[_key]['xywh']=xywh
        d_videos[_key]['filename']=filename
    return d_videos






#dir_path = '/data/TracKit/dataset/LaSOT/drone/drone-13/img/'
#img_list=[]
#for j in range(1,9999):
#    img_list.append('%08d' %j + '.jpg')

dir_path = '/data/dataset/shuttlecock/all/'
shuttlecock_dataset_path='/data/dataset/shuttlecock/'
shuttlecock_dataset = create_dataset_dict(shuttlecock_dataset_path)

print(shuttlecock_dataset.keys())
print(shuttlecock_dataset['1'].keys())
#print(shuttlecock_dataset['1']['filename'])
#print()
#print(shuttlecock_dataset['1'])
convert_to_video(dir_path,'shuttlecock_1.avi',shuttlecock_dataset['1']['filename'])





#/data/dataset/shuttlecock
#/data/dataset/shuttlecock/all/video_label_1_0_jpg.rf.ec4c1496209112f576dd0d0a91783569.jpg