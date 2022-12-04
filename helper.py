import numpy as np

def mean_error_IOD(fit_kp, gt_kp):

    err = np.zeros(gt_kp.shape[0])

    for i in range(gt_kp.shape[0]):
        fit_keypoints = fit_kp[i,:,:].squeeze()
        gt_keypoints = gt_kp[i, :, :].squeeze()
        face_error = 0
        for k in range(gt_kp.shape[1]):
            face_error += norm(fit_keypoints[k,:]-gt_keypoints[k,:]);
        face_error = face_error/gt_kp.shape[1];

        # pupil dis
        right_pupil = gt_keypoints[0, :];
        left_pupil = gt_keypoints[1, :];

        IOD = norm(right_pupil-left_pupil);

        if IOD != 0:
            err[i] = face_error/IOD
        else:
            print('IOD = 0!')

    return err.mean()
