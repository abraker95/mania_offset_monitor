import os
import sys
import time
import numpy as np
import tinydb

import pyqtgraph
from pyqtgraph import dockarea
from pyqtgraph.Qt import QtCore, QtGui

from PyQt5.QtGui import *

from osu_analysis import ManiaActionData, ManiaScoreData
from osu_analysis import BeatmapIO, ReplayIO, Gamemode
from osu_db_reader.osu_db_reader import OsuDbReader
from monitor import Monitor



class Data():    
    OFFSETS  = 0
    CURR_INT = 1
    PREV_INT = 2


class ManiaMonitor(QtGui.QMainWindow):

    new_replay_event = QtCore.pyqtSignal(str)

    def __init__(self, osu_path):
        QtGui.QMainWindow.__init__(self)

        os.makedirs('data', exist_ok=True)

        try: self.data_file = open('data/interval_offsets.npy', 'rb+')
        except FileNotFoundError:
            print('Data file not found. Creating...')

            np.save('data/interval_offsets', np.empty((0,3)), allow_pickle=False)
            self.data_file = open('data/interval_offsets.npy', 'rb+')

        self.data = np.load(self.data_file, allow_pickle=False)

        self.db = tinydb.TinyDB('data/maps.json')
        self.maps_table = self.db.table('maps')
        self.meta_table = self.db.table('meta')

        self.osu_path = osu_path
        self.__check_maps_db()
        self.__init_gui()

        if len(self.data) != 0:
            self.__process_data()
            self.__plot_data()

        self.monitor = Monitor(osu_path)
        self.monitor.create_replay_monitor('Replay Grapher', self.__handle_new_replay)

        self.new_replay_event.connect(self.__handle_new_replay_qt)

        self.show()


    def __del__(self):
        self.data_file.close()


    def __handle_new_replay(self, replay_path):
        time.sleep(1)
        self.new_replay_event.emit(replay_path)


    def __handle_new_replay_qt(self, replay_path):
        #print('New replay detected!')

        try: replay, beatmap = self.__get_files(replay_path)
        except TypeError: return

        try: self.setWindowTitle(beatmap.metadata.name + ' ' + replay.get_name())
        except AttributeError: pass

        map_data = ManiaActionData.get_action_data(beatmap)
        replay_data = ManiaActionData.get_action_data(replay)
        score_data = ManiaScoreData.get_score_data(map_data, replay_data)

        # Get data
        hit_note_intervals, hit_offsets, hit_timings = self.__get_hit_data(beatmap.difficulty.cs, map_data, score_data)
        #miss_note_intervals, miss_timings = self.__get_miss_data(beatmap.difficulty.cs, map_data, score_data)
        
        # Update data
        self.__update_data(hit_note_intervals, hit_offsets, hit_timings)

        # Process data
        self.__process_data()

        # Plot data
        self.__plot_data()


    def __check_maps_db(self):
        if len(self.maps_table) == 0:
            data = OsuDbReader.get_beatmap_md5_paths(f'{self.osu_path}/osu!.db')
            self.maps_table.insert_multiple(data)
            
            num_beatmaps_read = OsuDbReader.get_num_beatmaps(f'{self.osu_path}/osu!.db')
            self.meta_table.upsert({ 'num_maps' : num_beatmaps_read }, tinydb.where('num_maps').exists())

            last_modified_read = os.stat(f'{self.osu_path}/osu!.db').st_mtime
            self.meta_table.upsert({ 'last_modified' : last_modified_read }, tinydb.where('last_modified').exists())

            print('Map table did not exist - created it')
            return

        num_beatmaps_read = OsuDbReader.get_num_beatmaps(f'{self.osu_path}/osu!.db')
        num_beatmaps_save = self.meta_table.get(tinydb.where('num_maps').exists())
        if num_beatmaps_save != None:
            num_beatmaps_save = num_beatmaps_save['num_maps']

        last_modified_read = os.stat(f'{self.osu_path}/osu!.db').st_mtime
        last_modified_save = self.meta_table.get(tinydb.where('last_modified').exists())
        if last_modified_save != None:
            last_modified_save = last_modified_save['last_modified']

        num_maps_changed = num_beatmaps_read != num_beatmaps_save
        osu_db_modified = last_modified_read != last_modified_save

        if num_maps_changed or osu_db_modified:
            if osu_db_modified:
                user_input = input('osu!.db was modified. If you modified a map for testing, it will not be found until you rebuild db. Rebuild db? (y/n)')
                if not 'y' in user_input.lower(): return

            data = OsuDbReader.get_beatmap_md5_paths(f'{self.osu_path}/osu!.db')
            self.db.drop_table('maps')
            self.maps_table = self.db.table('maps')
            self.maps_table.insert_multiple(data)

            self.meta_table.upsert({ 'num_maps' : num_beatmaps_read }, tinydb.where('num_maps').exists())
            self.meta_table.upsert({ 'last_modified' : last_modified_read }, tinydb.where('last_modified').exists())

        print(num_beatmaps_read, num_beatmaps_save)
        print(last_modified_read, last_modified_save)


    def __init_gui(self):
        self.graphs = {}
        self.area = pyqtgraph.dockarea.DockArea()

        self.__create_graph(
            graph_id  = 'offset_interval',
            pos       = 'top',
            widget    = pyqtgraph.PlotWidget(title='Hits-Interval scatterplot'),
        )

        self.__create_graph(
            graph_id  = 'offset_interval2',
            pos       = 'bottom',
            widget    = pyqtgraph.PlotWidget(title='Hits-Interval2 scatterplot'),
        )

        self.graphs['offset_interval']['widget'].setLabel('left', 'Hit offset', units='ms', unitPrefix='')
        self.graphs['offset_interval']['widget'].setLabel('bottom', 'Interval', units='ms', unitPrefix='')

        self.avg_plot = self.graphs['offset_interval']['widget'].plot()
        
        self.std_plot_pos = self.graphs['offset_interval']['widget'].plot()
        self.std_plot_neg = self.graphs['offset_interval']['widget'].plot()
        self.graphs['offset_interval']['widget'].addItem(pyqtgraph.FillBetweenItem(self.std_plot_pos, self.std_plot_neg, (100, 100, 255, 100)))

        self.graphs['offset_interval2']['widget'].setLabel('left', 'Interval prev', units='ms', unitPrefix='')
        self.graphs['offset_interval2']['widget'].setLabel('bottom', 'Interval curr', units='ms', unitPrefix='')
 
        self.setCentralWidget(self.area)


    def __create_graph(self, graph_id=None, dock_name=' ', pos='bottom', relative_to=None, widget=None, plot=None):
        if type(widget) == type(None):
            widget = pyqtgraph.PlotWidget()
        
        try: widget.getViewBox().enableAutoRange()
        except AttributeError: pass
        
        dock = pyqtgraph.dockarea.Dock(dock_name, size=(500,400))
        dock.addWidget(widget)
        self.area.addDock(dock, pos, relativeTo=relative_to)

        self.graphs[graph_id] = {
            'widget' : widget,
            'dock'   : dock,
            'plot'   : widget.plot() if plot == None else plot
        }

        if plot != None:
            widget.addItem(plot)


    def __get_files(self, replay_path):
        #print('Loading replay...')

        try: replay = ReplayIO.open_replay(replay_path)
        except Exception as e:
            print(f'Error opening replay: {e}')
            return

        if replay.game_mode != Gamemode.MANIA:
            print('Only mania gamemode supported for now')            
            return

        #print(replay.mods)

        if replay.mods.has_mod('DT') or replay.mods.has_mod('NC'):
            print('DT and NC is not supported yet!')
            return 

        print('Determining beatmap...')

        maps = self.maps_table.search(tinydb.where('md5') == replay.beatmap_hash)
        if len(maps) == 0:
            print('Associated beatmap not found. Do you have it?')
            return

        path = f'{self.osu_path}/Songs'
        beatmap = BeatmapIO.open_beatmap(f'{self.osu_path}/Songs/{maps[0]["path"]}')

        return replay, beatmap


    def __get_hit_data(self, num_keys, map_data, score_data):
        note_intervals = []
        offsets        = []
        timings        = []

        for col in range(int(num_keys)):
            # Get note times where needed to press for current column
            map_filter = map_data[col] == ManiaActionData.PRESS
            map_times  = map_data.index[map_filter].values

            # Get scoring data for the current column
            score_col = score_data.loc[col]

            # Get score times associated with successful press for current column
            hit_filter    = score_col['type'] == ManiaScoreData.TYPE_HITP
            hit_map_times = score_col['map_t'][hit_filter].values

            # Correlate map's note press timings to scoring press times
            # Then get interval between the hit note and previous
            hit_time_filter = np.isin(map_times, hit_map_times)
            map_interval = np.diff(map_times)[hit_time_filter[1:]]

            # Get replay hitoffsets and timings for those offsets
            offset = (score_col['replay_t'] - score_col['map_t'])[hit_filter].values[1:]
            timing = score_col['replay_t'][hit_filter].values[1:]

            # Append data for column to overall data
            note_intervals.append(map_interval)
            offsets.append(offset)
            timings.append(timing)

            if len(offset) != len(map_interval):
                print(f'len(offsets) != len(note_intervals) for col {col}')
                print(f'len(map_times) = {len(map_times)}  len(hit_map_times) = {len(hit_map_times)}')
                print(f'len(hit_filter) = {len(hit_filter)}  len(hit_time_filter[1:]) = {len(hit_time_filter[1:])}')

                return [], [], []

        return np.concatenate(note_intervals), np.concatenate(offsets), np.concatenate(timings)


    def __update_data(self, hit_note_intervals, hit_offsets, hit_timings):
        # Sort the hits by timing
        sort_map = np.argsort(hit_timings)
        
        hit_note_intervals = hit_note_intervals[sort_map]
        hit_offsets = hit_offsets[sort_map]

        # TODO:
        # [ 
        #   ic0, ic1, ic2, ... ic18,         # note offset from current hit to prev note of each column
        #   ip0, ip1, ip2, ... ip18,         # note offset from prev hit to prev prev note of each column
        #   h0, h1, h2, ... h18,             # hold state for each column at release timing
        #   hit_offset, release_offset       # hit timing
        #   keys, timestamp, map_id          # metadata
        # ]

        # Create data to save
        data = np.asarray([
            hit_offsets[1:],
            hit_note_intervals[1:],
            hit_note_intervals[:-1]
        ]).transpose()

        # Queue for save
        self.data = np.insert(self.data, 0, data, axis=0)
        np.save(self.data_file, self.data, allow_pickle=False)

        # Actually saves to disk
        self.data_file.close()

        # Now reopen it so it can be used
        self.data_file = open('data/interval_offsets.npy', 'rb+')


    def __process_data(self):
        xMax = max(self.data[:, Data.CURR_INT])
        xMin = min(self.data[:, Data.CURR_INT])
        
        num_bins = 300
        bin_width = (xMax - xMin)/num_bins
        
        bins = np.linspace(xMin, xMax, num_bins, endpoint=True)  # Create bin ranges
        idxs = np.digitize(self.data[:, Data.CURR_INT], bins)    # Get indices for each bin range (start with 1)
        
        offset_avgs = np.asarray([ self.data[:, Data.OFFSETS][idxs == i].mean() if len(self.data[:, Data.OFFSETS][idxs == i]) > 20 else 0 for i in range(1, len(bins)) ])
        offset_stddev = np.asarray([ self.data[:, Data.OFFSETS][idxs == i].std() if len(self.data[:, Data.OFFSETS][idxs == i]) > 20 else 0 for i in range(1, len(bins)) ])

        self.avg_plot.setData(bins[:-1] + bin_width/2, offset_avgs, pen='y')
        self.std_plot_pos.setData(bins[:-1] + bin_width/2, offset_stddev*2 + offset_avgs, pen='k')
        self.std_plot_neg.setData(bins[:-1] + bin_width/2, -offset_stddev*2 + offset_avgs, pen='k')


    def __plot_data(self):
        xMax = max(self.data[:, Data.CURR_INT]) + 100

        # Drawing optimization: filter out duplicate points
        unique = np.unique(self.data[:, [Data.CURR_INT, Data.OFFSETS]], axis=0)

        # Plot
        self.graphs['offset_interval']['plot'].setData(unique[:, 0], unique[:, 1], pen=None, symbol='o', symbolPen=None, symbolSize=2, symbolBrush=(100, 100, 255, 200))
        self.graphs['offset_interval']['widget'].setLimits(xMin=0, xMax=xMax, yMin=-120, yMax=120)

        symbolBrush = pyqtgraph.ColorMap(
            np.array([ 0, 16, 50, 100 ]),  # Offsets
            np.array([                     # Colors
                [ 0,   255,    0, 10   ],
                [ 0,   0,    0, 50  ],
                [ 255, 255,  0, 100 ],
                [ 255, 0,    0, 200 ],
            ]),
        ).map(np.abs(self.data[:, Data.OFFSETS]), 'qcolor')

        self.graphs['offset_interval2']['plot'].setData(self.data[:, Data.CURR_INT], self.data[:, Data.PREV_INT], pen=None, symbol='o', symbolPen=None, symbolSize=2, symbolBrush=symbolBrush)
        self.graphs['offset_interval2']['widget'].setLimits(xMin=0, xMax=900, yMin=0, yMax=900)

        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex  = ManiaMonitor('C:/Games/osu!')
    sys.exit(app.exec_())
