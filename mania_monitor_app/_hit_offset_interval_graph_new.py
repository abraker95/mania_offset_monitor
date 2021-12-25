from numpy.core.fromnumeric import shape
import pyqtgraph
from pyqtgraph import dockarea
from pyqtgraph.Qt import QtCore, QtGui

from PyQt5.QtGui import *
import numpy as np

from osu_analysis import ManiaScoreData
from osu_performance_recorder import Data

from ._utils import Utils
from ._callback import callback


class HitOffsetIntervalGraphNew():

    @callback
    class model_updated_event(): pass

    def __init__(self, pos, relative_to=None):

        self.__id = __class__.__name__
        self._create_graph(
            graph_id    = self.__id,
            pos         = pos,
            relative_to = relative_to,
            widget      = pyqtgraph.PlotWidget(title='Average Hit Offset for Intervals (New)'),
        )

        self.graphs[self.__id]['widget'].getPlotItem().getAxis('left').enableAutoSIPrefix(False)
        self.graphs[self.__id]['widget'].getPlotItem().getAxis('bottom').enableAutoSIPrefix(False)
        self.graphs[self.__id]['widget'].setLabel('left', 'Avg hit offset', units='ms', unitPrefix='')
        self.graphs[self.__id]['widget'].setLabel('bottom', 'Note interval', units='ms', unitPrefix='')

        self.__model_plot = self.graphs[self.__id]['widget'].plot()

        self.__model_std_line_x_pos = pyqtgraph.InfiniteLine(angle=0, pen=pyqtgraph.mkPen((255, 150, 0, 150), width=1))
        self.__model_std_line_x_neg = pyqtgraph.InfiniteLine(angle=0, pen=pyqtgraph.mkPen((255, 150, 0, 150), width=1))

        self.__model_std_line_y_pos = pyqtgraph.InfiniteLine(angle=90, pen=pyqtgraph.mkPen((255, 150, 0, 150), width=1))
        self.__model_std_line_y_neg = pyqtgraph.InfiniteLine(angle=90, pen=pyqtgraph.mkPen((255, 150, 0, 150), width=1))
        self.__model_avg_line_x = pyqtgraph.InfiniteLine(angle=90, pen=pyqtgraph.mkPen((150, 255, 0, 150), width=1))

        self.graphs[self.__id]['widget'].addItem(self.__model_std_line_x_pos)
        self.graphs[self.__id]['widget'].addItem(self.__model_std_line_x_neg)
        #self.graphs[self.__id]['widget'].addItem(self.__model_std_line_y_pos)
        #self.graphs[self.__id]['widget'].addItem(self.__model_std_line_y_neg)
        #self.graphs[self.__id]['widget'].addItem(self.__model_avg_line_x)
        self.graphs[self.__id]['widget'].setLimits(yMin=-1)
        

    def _plot_data(self, data, time_range=None):
        HitOffsetIntervalGraphNew.__plot_offset_intervals(self, data, time_range)
        HitOffsetIntervalGraphNew.__plot_model(self, data)
        

    def __plot_offset_intervals(self, data, time_range=None):
        half_window_width = 10
        min_interval_peak = 100
        unique_timestamps = np.unique(data[:, Data.TIMESTAMP])

        if type(time_range) != type(None):
            start, end = time_range
            unique_timestamps = unique_timestamps[(start <= unique_timestamps) & (unique_timestamps <= end)]
        
        x_note_intervals  = []
        y_delta_intervals = []
        timestamps        = []

        # Got through the list of plays recorded
        for timestamp in unique_timestamps:
            # Get score data for the play
            data_filter = \
                (data[:, Data.TIMESTAMP] == timestamp) & \
                (data[:, Data.HIT_TYPE] == ManiaScoreData.TYPE_HITP)
            data_slice = data[data_filter]

            # Used to determine note of notes in column and select note in column
            col1_select = data_slice[:, Data.KEYS] == 0
            col2_select = data_slice[:, Data.KEYS] == 1
            col3_select = data_slice[:, Data.KEYS] == 2
            col4_select = data_slice[:, Data.KEYS] == 3

            num_notes_cols = np.array([
                [ np.count_nonzero(col1_select), col1_select ],
                [ np.count_nonzero(col2_select), col2_select ],
                [ np.count_nonzero(col3_select), col3_select ],
                [ np.count_nonzero(col4_select), col4_select ],
            ], dtype=object)

            # Allocate array to store data in
            data_size = np.sum(num_notes_cols[:, 0]) - num_notes_cols.shape[0]
            note_intervals  = np.zeros(data_size)
            delta_intervals = np.zeros(data_size)

            idx = 0

            # Calculate interval data per-column
            for i in range(num_notes_cols.shape[0]):
                num_notes = num_notes_cols[i][0] - 1
                col_select = num_notes_cols[i][1]

                if num_notes <= 0:
                    continue

                hit_timings = data_slice[col_select, Data.TIMINGS]
                hit_offsets = data_slice[col_select, Data.OFFSETS]
                note_timings = hit_timings - hit_offsets
                col_intervals = np.diff(note_timings)

                note_intervals[idx : idx+num_notes] = col_intervals
                delta_intervals[idx : idx+num_notes] = np.diff(hit_timings) - col_intervals
                
                idx += num_notes

            # Filter out note intervals that don't occur frequently enough
            interv_freqs = Utils.get_freq_hist(note_intervals)
            note_intervals = note_intervals[interv_freqs >= min_interval_peak]
            delta_intervals = delta_intervals[interv_freqs >= min_interval_peak]
            
            unique_note_intervals = np.unique(note_intervals)

            # Collect offsets in relation to peaks
            for unique_note_interval in unique_note_intervals:
                intervals = delta_intervals[note_intervals == unique_note_interval]
                
                y_delta_intervals.append(np.mean(intervals))
                x_note_intervals.append(unique_note_interval)
                timestamps.append(timestamp)

        if len(x_note_intervals) == 0:
            return

        # Calculate view
        xMin = min(x_note_intervals) - 100
        xMax = max(x_note_intervals) + 100
        yMax = max(y_delta_intervals) + 10
        yMin = min(y_delta_intervals) - 10

        # Calculate color
        is_latest   = max(data[:, Data.TIMESTAMP]) == np.asarray(timestamps)
        symbol_size = is_latest*2 + 2

        symbolBrush = pyqtgraph.ColorMap(
            np.array([ 0, 1 ]),            # Is latest?
            np.array([                     # Colors
                [ 100, 100, 255, 200  ],
                [ 0,   255, 0,   255  ],
            ]),
        ).map(is_latest, 'qcolor')

        # Cache for plotting model
        self.__x_note_intervals = np.asarray(x_note_intervals)
        self.__y_mean_offsets   = np.asarray(y_delta_intervals)

        # Plot data
        self.graphs[self.__id]['plot'].setData(x_note_intervals, y_delta_intervals, pen=None, symbol='o', symbolPen=None, symbolSize=symbol_size, symbolBrush=symbolBrush)
        self.graphs[self.__id]['widget'].setLimits(xMin=xMin, yMin=yMin, xMax=xMax, yMax=yMax)


    def __plot_model(self, data):
        # Determine the y offset by getting the average mean offset within 16 ms range (60 fps)
        hor_area = self.__y_mean_offsets[(-16 <= self.__y_mean_offsets) & (self.__y_mean_offsets < 16)]
        p0y = np.average(hor_area)

        # Standard deviation of the points in the non straining region
        # This is used to center the model since the points determined initially are on the edge
        p0y_std = np.std(hor_area)

        top_std = p0y_std*2 + p0y
        btm_std = -p0y_std*2 + p0y
        '''
        # Get area of points above the 2*std line
        top_area_x = self.__x_note_intervals[top_std <= self.__y_mean_offsets]
        top_area_y = self.__y_mean_offsets[top_std <= self.__y_mean_offsets]

        # Get center of top area
        p1x = np.average(top_area_x, weights=1/top_area_x**2)
        p1y = np.average(top_area_y)

        # Get stdev of top area on vertical axis
        p1x_std = np.std(top_area_x)
        lft_std = -p1x_std + p1x
        rgt_std = p1x_std + p1x

        # Get area of points with area bounded by the 2*std_x and 2*std_y lines
        lft_area_x = self.__x_note_intervals[(lft_std < self.__x_note_intervals) & (self.__x_note_intervals < rgt_std)]
        lft_area_y = self.__y_mean_offsets[(btm_std < self.__y_mean_offsets) & (self.__y_mean_offsets < top_std)]
        
        # Get center of left area
        p2x = np.average(lft_area_x)
        print(lft_std, rgt_std)
        p2y = np.average(lft_area_y)

        # Get slope (p2x, p2y) to (p1x, p1y)
        r = (p2y - p1y)/(p2x - p1x)

        # tmin is at p2x
        t_min = p2x
        '''

        # Get a region of points between 0 and 2 note interval
        # Find the point that is left most in that region
        # Then shift from left towards center of data using stdev
        p0x = min(self.__x_note_intervals[(0 <= self.__y_mean_offsets) & (self.__y_mean_offsets < 2)]) # + 2*p0y_std
        
        # Get a region of point that are greater than 0 mean offset
        # Find the point that is left most in that region and top most in that region.
        # Then shift from top towards center of data using stdev
        p1x = min(self.__x_note_intervals[0 <= self.__x_note_intervals])
        p1y = max(self.__y_mean_offsets[0 <= self.__x_note_intervals]) #- 2*p0y_std

        r = (p0y - p1y)/(p0x - p1x)
        t_min = p0x

        err = Utils.calc_err(self.__x_note_intervals, self.__y_mean_offsets, r, t_min, p0y)/len(self.__x_note_intervals)
        print(f'r = {r:.2f}   t_min = {t_min:.2f} ms ({(1000*60)/(t_min*2):.2f} bpm)  y = {p0y:.2f} ms  err = {err:.4f}')
        curve_fit = Utils.softplus_func(self.__x_note_intervals, r, t_min, p0y)

        idx_sort = np.argsort(self.__x_note_intervals)
        self.__model_plot.setData(self.__x_note_intervals[idx_sort], curve_fit[idx_sort], pen='y')

        self.__model_std_line_x_pos.setValue(top_std)
        self.__model_std_line_x_neg.setValue(btm_std)
        '''
        self.__model_std_line_y_pos.setValue(lft_std)
        self.__model_std_line_y_neg.setValue(rgt_std)
        self.__model_avg_line_x.setValue(p1x)
        '''

        HitOffsetIntervalGraphNew.model_updated_event.emit((r, t_min, p0y))