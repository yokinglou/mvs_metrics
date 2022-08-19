import os
from utils import * 
from metrics_np import * 

def evaluate_pc():
    est_pc_path = 'data/640_512/single_view_pc_est'
    gt_pc_path = 'data/640_512/single_view_pc_gt'
    est_pc_lists = os.listdir(est_pc_path)
    gt_pc_lists = os.listdir(gt_pc_path)
    est_pc_lists.sort()
    gt_pc_lists.sort()

    results = []
    for i in range(len(est_pc_lists)):
        print ('evaluating {}...'.format(i))
        R_ply_path = os.path.join(est_pc_path, est_pc_lists[i])
        G_ply_path = os.path.join(gt_pc_path, gt_pc_lists[i])
        R = read_ply(R_ply_path)
        G = read_ply(G_ply_path)
        result = single_view_pc_evaluation(R, G)
        results.append(result)
        
    results = np.array(results)
    np.save('results/pc_results.npy', results)
    mean_results = np.mean(results, axis=0)
    
    print ("Distance Metrics:")
    print ("Accuracy:\t{}".format(mean_results[0]))
    print ("Completeness:\t{}".format(mean_results[1]))
    print ("Overall:\t{}".format(mean_results[2]))
    print ("Percentage Metrics:")
    print ("Accuracy:\t{}".format(mean_results[3]))
    print ("Completeness:\t{}".format(mean_results[4]))
    print ("F-Score:\t{}".format(mean_results[5]))
    print ("Normals 2-Norm:\t{}".format(mean_results[6]))

def evaluate_depth():
    dep_est_paths = ['data/640_512/depth_est',
                    'data/640_512/depth_est_with_final_mask',
                    'data/640_512/depth_est_with_geo_mask',
                    'data/640_512/depth_est_with_photo_mask']
    
    for k in range(len(dep_est_paths)):
        dep_est_path = dep_est_paths[k]
        depth_gt_path = 'data/640_512/depth_gt'

        dep_est_list = os.listdir(dep_est_path)
        depth_gt_list = os.listdir(depth_gt_path)
        dep_est_list.sort()
        depth_gt_list.sort()
                
        results = []
        for i in dep_est_list:
            est, _ = np.asarray(read_pfm(os.path.join(dep_est_path, i)))

            if np.isnan(est).sum() > 0:
                print ('est')
                print (np.isnan(est).sum())
            gt, _ = read_pfm(os.path.join(depth_gt_path, i))

            if np.isnan(gt).sum() > 0:
                continue
                # print (i)
                # print ('gt')
            result = single_view_depth_evaluation(est[:, :, 0], gt[:, :, 0])
            
            results.append(result)
        results = np.array(results)
        mean_results = np.mean(results, axis=0)
        np.save('results/depth_results{}.npy'.format(k), results)

        print ('Mean results {}:'.format(k))
        print ("MSE: {}".format(mean_results[0]))
        print ("RMSE: {}".format(mean_results[1]))
        print ("PSNR: {}".format(mean_results[2]))
        print ("SSIM: {}".format(mean_results[3]))
        print ("EPE: {}".format(mean_results[4]))    

if __name__ == '__main__':
    evaluate_pc()
    evaluate_depth()

    