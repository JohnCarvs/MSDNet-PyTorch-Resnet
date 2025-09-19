python main.py --data-root data/imagenet --data ImageNet --save ./save --arch msdnet --batch-size 32 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --evalmode dynamic --evaluate-from ./save/save_models/msdnet-step=7-block=5.pth.tar --use-valid --gpu 0 -j 8




python main.py --data-root data/cifar --data cifar100 --save save --arch msdnet --batch-size 64 --epochs 300 --nBlocks 7 --stepmode even --step 2 --base 4 --nChannels 16 -j 4          




python main.py --data-root data/imagenetc --data ImageNet --save save --arch msdnet --batch-size 4096 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --evalmode dynamic --evaluate-from ./save/save_models/msdnet-step=4-block=5.pth.tar --use-valid --gpu 1,2 -j 64 
python main_inference_imagenetc_bayes_voting.py --data-root data/imagenetc --data ImageNet --save save --arch msdnet --batch-size 4096 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --evalmode dynamic --evaluate-from ./save/save_models/msdnet-step=4-block=5.pth.tar --use-valid --gpu 1 -j 64 

# iterate for each corruption config
python main_inference_imagenetc_bayes_voting_iterate_corruption_config.py --data-root data/imagenetc --data ImageNet --save save --arch msdnet --batch-size 4096 --epochs 90 --nBlocks 5 --stepmode even --step 4 --base 4 --nChannels 32 --growthRate 16 --grFactor 1-2-4-4 --bnFactor 1-2-4-4 --evalmode dynamic --evaluate-from ./save/save_models/msdnet-step=4-block=5.pth.tar --use-valid --gpu 1,2 -j 64 > output.txt