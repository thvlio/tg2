# 1205a

# extract data from the bags
python extract_data.py --bag bags/1205n/train1.bag --dest 1205a --set "train"
python extract_data.py --bag bags/1205n/train2.bag --dest 1205a --set "train" --append
python extract_data.py --bag bags/1205n/test.bag --dest 1205a --set "test"
python extract_data.py --bag bags/1205n/val.bag --dest 1205a --set "val"

# 1205b

# extract data from the bags
# python extract_data.py --bag bags/1205n/train1.bag --dest 1205b --set "train"
# python extract_data.py --bag bags/1205n/train2.bag --dest 1205b --set "train" --append
# python extract_data.py --bag bags/1205p/train1.bag --dest 1205b --set "train" --append
# python extract_data.py --bag bags/1205p/train2.bag --dest 1205b --set "train" --append
# python extract_data.py --bag bags/1205n/test.bag --dest 1205b --set "test"
# python extract_data.py --bag bags/1205p/test.bag --dest 1205b --set "test" --append
# python extract_data.py --bag bags/1205n/val.bag --dest 1205b --set "val"
# python extract_data.py --bag bags/1205p/val.bag --dest 1205b --set "val" --append

# 1205c

# extract data from the bags
python extract_data.py --bag bags/1205e/train1.bag --dest 1205c --set "train"
python extract_data.py --bag bags/1205e/train2.bag --dest 1205c --set "train" --append
python extract_data.py --bag bags/1205e/test.bag --dest 1205c --set "test"
python extract_data.py --bag bags/1205e/val.bag --dest 1205c --set "val"

# 1205d

# extract data from the bags
# python extract_data.py --bag bags/1205e/train1.bag --dest 1205d --set "train"
# python extract_data.py --bag bags/1205e/train2.bag --dest 1205d --set "train" --append
# python extract_data.py --bag bags/1205p/train1.bag --dest 1205d --set "train" --append
# python extract_data.py --bag bags/1205p/train2.bag --dest 1205d --set "train" --append
# python extract_data.py --bag bags/1205e/test.bag --dest 1205d --set "test"
# python extract_data.py --bag bags/1205p/test.bag --dest 1205d --set "test" --append
# python extract_data.py --bag bags/1205e/val.bag --dest 1205d --set "val"
# python extract_data.py --bag bags/1205p/val.bag --dest 1205d --set "val" --append
