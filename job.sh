# pip install -r requirements.txt
# pip install -r _requirements.txt
cd ./ner/


# python3.8 ./data_processing.py --choosen_dataset processed_1 # > ./result/event/log.txt
for i in 1 # 2 3 4 5 6 7 8
do
    python3 ./gruscanet.py --choosen_dataset processed_$i >> ./result/event/log.txt
done