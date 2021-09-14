#!/usr/bin/env python
# -- coding: utf-8 --

"""
Hannah weiser, September 2021
h.weiser@stud.uni-heidelberg.de
"""

import cv2
from pathlib import Path, PurePath
import pandas as pd
import PTV_functions as ptv
import numpy as np
import featureReference_functions as refF
import draw_functions as drawF
import featureFilter_functions as filterF
import photogrammetry_functions as photogrF
import matplotlib.pyplot as plt
import matplotlib


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:

        # writing filename to output file
        print(str(img_path))
        outf.write(str(img_path) + " 1")

        # displaying the coordinates
        # a) on the Shell
        print(x, ' ', y)

        # b) on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.drawMarker(img, (x, y), (255, 0, 0), markerType=cv2.MARKER_SQUARE, markerSize=10)
        cv2.imshow('image', img)

        # write image coordinates to file
        outf.write(' %i %i\n' % (x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:

        # writing filename to output file
        print(str(img_path))
        outf.write(str(img_path) + " 2")

        # displaying the coordinates
        # a) on the Shell
        print(x, ' ', y)

        # b) on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 255, 0), 2)
        cv2.drawMarker(img, (x, y), (255, 0, 0), markerType=cv2.MARKER_SQUARE, markerSize=10)
        cv2.imshow('image', img)

        # write image coordinates to file
        outf.write(' %i %i\n' % (x, y))


def TracksPx_to_TracksMetric(filteredFeatures, interior_orient, eor_mat, unit_gcp,
                             frame_rate_cam, TrackEveryNthFrame, waterlevel_pt,
                             directoryOutput, img_name, every_xth=1):
    # scale tracks in image space to tracks in object space to get flow velocity in m/s
    waterlevel = waterlevel_pt
    filteredFeatures_count = np.asarray(filteredFeatures.groupby('id', as_index=False).count())[:, 2]

    filteredFeatures_list = []
    if every_xth > 1:
        for group, coords in filteredFeatures.groupby('id', as_index=False):
            for i, j in enumerate(range(0, coords.shape[0]-every_xth+1, every_xth)):
                filteredFeatures = coords.iloc[j:j + every_xth + 1]
                if group == 1:
                    filteredFeatures_list.append(filteredFeatures)
                else:
                    df = filteredFeatures_list[i]
                    df = df.append(filteredFeatures)
                    filteredFeatures_list[i] = df
    else:
        filteredFeatures_list = [filteredFeatures]

    subtracks = len(filteredFeatures_list)

    for k, filteredFeatures in enumerate(filteredFeatures_list):
        image = cv2.imread(img_name, 0)

        print(filteredFeatures)
        # get first and last position in image space of each tracked feature
        filteredFeatures_1st = filteredFeatures.groupby('id', as_index=False).head(1)
        filteredFeatures_last = filteredFeatures.groupby('id', as_index=False).tail(1)
        filteredFeatures_count = np.asarray(filteredFeatures.groupby('id', as_index=False).count())[:, 2]
        print("count", filteredFeatures_count)

        xy_start_tr = np.asarray(filteredFeatures_1st[['x', 'y']])
        xy_tr = np.asarray(filteredFeatures_last[['x', 'y']])

        # intersect first and last position with waterlevel
        XY_start_tr = refF.LinePlaneIntersect(xy_start_tr, waterlevel, interior_orient, eor_mat, unit_gcp) / unit_gcp
        XY_tr = refF.LinePlaneIntersect(xy_tr, waterlevel, interior_orient, eor_mat, unit_gcp) / unit_gcp

        # get angle of track
        x_track = xy_tr[:, 0] - xy_start_tr[:, 0]
        y_track = xy_tr[:, 1] - xy_start_tr[:, 1]
        track = np.hstack((x_track.reshape(x_track.shape[0], 1), y_track.reshape(y_track.shape[0], 1)))
        angle = np.degrees(filterF.angleBetweenVecAndXaxis(track))

        # get corresponding distance in object space
        dist_metric = np.sqrt(np.square(XY_start_tr[:, 0] - XY_tr[:, 0]) + (np.square(XY_start_tr[:, 1] - XY_tr[:, 1])))

        # get corresponding temporal observation span
        frame_rate = np.ones((filteredFeatures_count.shape[0], 1), dtype=np.float) * np.float(frame_rate_cam)
        nbrTrackedFrames = TrackEveryNthFrame * (filteredFeatures_count-1)
        trackingDuration = nbrTrackedFrames.reshape(frame_rate.shape[0], 1) / frame_rate
        print(trackingDuration)

        # get velocity
        velo = dist_metric.reshape(trackingDuration.shape[0], 1) / trackingDuration
        filteredFeatures_1st = ptv.filterFeatureOrganise(filteredFeatures_1st, XY_start_tr, XY_tr, xy_tr, dist_metric,
                                                         velo, True, filteredFeatures_count-1)
        filteredFeatures = filteredFeatures_1st.copy()
        filteredFeatures = filteredFeatures.reset_index(drop=True)
        filteredFeaturesRawPTVOut = filteredFeatures[['X', 'Y', 'Z', 'velo', 'dist_metric', 'count']]
        filteredFeaturesRawPTVOut.columns = ['X', 'Y', 'Z', 'velo', 'dist', 'count']
        filteredFeaturesRawPTVOut['angle'] = angle.values
        filteredFeaturesRawPTVOut['duration'] = filteredFeaturesRawPTVOut['count'] * TrackEveryNthFrame
        if subtracks == 1:
            suffix = ''
        else:
            suffix = '_sub%s' % k
        filteredFeaturesRawPTVOut.to_csv(directoryOutput + 'TracksReferenced_raw_PTV' + suffix + '.txt', sep='\t',
                                         index=False)
        del filteredFeaturesRawPTVOut
        draw_tracks(filteredFeatures, image, directoryOutput, 'TracksReferenced_raw_PTV' + suffix + '.jpg',
                          'velo', colors=["red", "deepskyblue"], label_data="True", variableToLabel='velo')

        print('nbr of tracked features: ' + str(filteredFeatures.shape[0]) + '\n')


def draw_tracks(Final_Vals, image, dir_out, outputImgName, variableToDraw, colors,
                label_data=False, variableToLabel=None):
    try:
        '''visualize'''
        # sort after flow velocity
        image_points = Final_Vals.sort_values(variableToDraw)
        image_points = image_points.reset_index(drop=True)

        # set font size
        fontProperties_text = {'size' : 12,
                               'family' : 'serif'}
        matplotlib.rc('font', **fontProperties_text)

        # draw figure
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.axis('equal')
        fig.add_axes(ax)

        image_points = image_points.sort_values('id')
        image_points = image_points.reset_index(drop=True)

        # add arrows
        point_n = 0
        label_criteria = 0

        while point_n <= image_points.shape[0]:
            try:
                if label_data:
                    id, label, xl, yl, arr_x, arr_y = image_points['id'][point_n], image_points[variableToLabel][point_n], image_points['x'][point_n], image_points['y'][point_n], image_points['x_tr'][point_n], image_points['y_tr'][point_n]
                else:
                    xl, yl, arr_x, arr_y = image_points['x'][point_n], image_points['y'][point_n], image_points['x_tr'][point_n], image_points['y_tr'][point_n]
                ax.arrow(xl, yl, arr_x-xl, arr_y-yl, color=colors[id-1],
                         head_width=5, head_length=5, width=1.5)

                if label_data:
                    if id == 1:
                        ax.annotate(str("{0:.2f}".format(label)), xy=(xl+25, yl+25), color=colors[id-1],
                                    **fontProperties_text)
                    else:
                        ax.annotate(str("{0:.2f}".format(label)), xy=(xl+35, yl-25), color=colors[id-1],
                                    **fontProperties_text)
                point_n += 1

            except Exception as e:
                point_n += 1

        ax.imshow(image, cmap='gray')

        #save figure
        plt.savefig(str(Path(dir_out) / outputImgName),  dpi=600)
        plt.close('all')
        plt.clf()

    except Exception as e:
        print(e)


# driver function
if __name__ == "__main__":

    images = Path(r'I:\UAV-photo\Befliegung_2020\for_velocity\frames_30fps_coreg').glob('frame*_coreg.jpg')
    outfile = PurePath(r'I:\UAV-photo\Befliegung_2020\for_velocity', 'branch_coords2.txt')


    outf = open(outfile, 'w')
    outf.write("filename id x y\n")

    for i, img_path in enumerate(images):
        if i % 30 == 0:  # only every 30th image -> tracking stem every second
            # reading the image
            img = cv2.imread(str(img_path))

            # resizing display window
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.moveWindow('image', 10, 10)
            cv2.resizeWindow('image', 1600, 900)

            # displaying the image
            cv2.imshow('image', img)

            # setting mouse handler for the image
            # and calling the click_event() function
            cv2.setMouseCallback('image', click_event)

            # wait for a key to be pressed to exit
            cv2.waitKey(0)

            # close the window
            cv2.destroyAllWindows()

    outf.close()


    last_img = r'I:\UAV-photo\Befliegung_2020\for_velocity\frames_30fps_coreg\frame00331_coreg.jpg'
    img = cv2.imread(last_img)

    cv2.namedWindow('track', cv2.WINDOW_NORMAL)
    cv2.moveWindow('track', 10, 10)
    cv2.resizeWindow('track', 1600, 900)

    # create track and write onto last image
    df_coords = pd.read_csv(outfile, sep=" ")
    df_coords_1 = df_coords[df_coords.id == 1]
    df_coords_2 = df_coords[df_coords.id == 2]

    for i, vals in enumerate(df_coords_1.values):
        if i < 1:
            continue
        else:
            prev_point = df_coords_1.values[i-1]
        cv2.line(img, (prev_point[2], prev_point[3]), (vals[2], vals[3]), color=(0, 0, 255), thickness=2)

    for i, vals in enumerate(df_coords_2.values):
        if i < 1:
            continue
        else:
            prev_point = df_coords_2.values[i-1]
        cv2.line(img, (prev_point[2], prev_point[3]), (vals[2], vals[3]), color=(255, 191, 0), thickness=2)

    out_dir = Path(r'I:\UAV-photo\FlowVeloTool_Acc_using_branch') / outfile.parts[-1].replace(".txt", "")
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")
    cv2.imshow('track', img)
    cv2.imwrite(str(Path(out_dir) / "branch_track.jpg"), img)
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()

###
    ior_file = r'I:\UAV-photo\Befliegung_2020\for_velocity\sensorInteriorOrientation.txt'
    gcpCoo_file = r"I:\UAV-photo\Befliegung_2020\for_velocity\GCPsinObjectSpace.txt"
    imgCoo_GCP_file = r"I:\UAV-photo\Befliegung_2020\for_velocity\GCPsInImage.txt"
    interior_orient = photogrF.read_aicon_ior(ior_file)
    print("Computing velocity")
    eor_mat = ptv.EstimateExterior(gcpCoo_file, imgCoo_GCP_file, interior_orient, estimate_exterior=True,
                                   unit_gcp=1000.0, max_orientation_deviation=1, ransacApprox=True, angles_eor=None,
                                   pos_eor=None, directoryOutput=str(out_dir))
    TracksPx_to_TracksMetric(df_coords, interior_orient, eor_mat, unit_gcp=1000.0,
                             frame_rate_cam=30, TrackEveryNthFrame=30, waterlevel_pt=137.0,
                             directoryOutput=str(out_dir), img_name=last_img, every_xth=1)
    TracksPx_to_TracksMetric(df_coords, interior_orient, eor_mat, unit_gcp=1000.0,
                             frame_rate_cam=30, TrackEveryNthFrame=30, waterlevel_pt=137.0,
                             directoryOutput=str(out_dir), img_name=last_img, every_xth=4)
