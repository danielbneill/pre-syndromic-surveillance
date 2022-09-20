from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import time
import numpy as np
import uuid
import argparse

parser = argparse.ArgumentParser(description='Visualization args parser')

# Data files
parser.add_argument('--results_folder', type=str, required = True)
parser.add_argument('--monitored_topic_file', type=str, required = True)

FLAGS = parser.parse_args()

# Represents a disease cluster; contains basic info and a list to the cases invovled.
class Cluster:
    index = 0
    score = 0
    startdate = ''
    enddate = ''
    words = ''
    locations = ''
    ages = ''
    monitor = 0
    normalized = 0

    words_list = []
    case_list = []

    def __init__(
            self, index, score, startdate, enddate, words,
            locations, ages, monitor=0, normalized=0
    ):
        self.index = int(index)
        self.score = round(float(score), 5)
        self.startdate = startdate
        self.enddate = enddate
        self.words = words
        self.locations = locations
        self.ages = ages
        self.monitor = monitor
        self.normalized = normalized
        self.words_list = []
        self.case_list = []

    def __str__(self):
        return str(self.to_tuple())

    def __repr__(self):
        return str(self)

    def to_tuple(self):
        return self.score, self.startdate, self.enddate, self.words, self.locations, self.ages

# Represents a single case that belongs to a certain cluster. Referenced by Cluster class.
class Case:
    cluster_index = 0
    date = ''
    time = ''
    location = ''
    words = ''
    icd = ''
    sex = ''
    age = ''
    id = ''
    uuid = -1

    def __init__(
            self, cluster_index, date, time, location, words, icd, sex, age, id, uuid
    ):
        self.cluster_index = int(cluster_index)
        self.date = date
        self.time = time
        self.location = location
        self.words = words
        self.icd = icd
        self.sex = sex
        self.age = age
        self.id = id
        self.uuid = uuid

    def __str__(self):
        return str(self.to_tuple())

    def __repr__(self):
        return str(self)

    def to_tuple(self):
        return self.date, self.time, self.location, self.words, self.icd, self.sex, self.age, self.id

# Helper function for determining if a number is float or not

def isfloat(num):
    try:
        a = float(num)
    except:
        return False
    return True

# Helper function for inserting cluster rows into the table
def insert_rows(t, l):
    for cluster in l.values():
        t.insert(
            parent='',
            index='end',
            iid=cluster.index,
            values=cluster.to_tuple())

# Open the file selection GUI and select the corresponding file
# filetype: the file/folder that needs to be selected

def select_file(file_type):
    if file_type == 0 or file_type == 'folder':
        ret = filedialog.askdirectory(
            title='Select Result Folder',
        )
        return ret
    elif file_type == 1 or file_type == 'cluster':
        ret = filedialog.askopenfilename(
            title='Select Static Topic File',
            filetypes=(('CSV File', '*.csv'), ("All Files", "."))
        )
        return ret
    elif file_type == 2 or file_type == 'setting':
        ret = filedialog.askopenfilename(
            title='Select Visualization Setting File',
            filetypes=(('Text File', '*.txt'), ("All Files", "."))
        )
        return ret
    return ''

# Read the settings of the program (the location of the cluster folder and the topic folder)

def read_settings(settings_file_name):
    try:
        settings = open(settings_file_name, 'r')
    except:
        messagebox.showinfo(
            title='Error',
            message='Loading Setting File Failed!')
        return '', ''

    contents = settings.readlines()
    settings.close()

    # Begin processing folders and file

    result_folder_name = ''
    static_topics_file_name = ''

    # Create new setting file if necesary
    if len(contents) != 2:
        messagebox.showinfo(
            title='Error',
            message='Visualization setting file not right! \
             Please reselect result folder and cluster file')
        static_topics_file_name = select_file('cluster')
        result_folder_name = select_file('folder')
        return result_folder_name, static_topics_file_name

    try:
        if contents[1][0:19] != 'static_topics_file,':
            raise ValueError
        static_topics_file_name = contents[1][19:]
        if not os.path.exists(static_topics_file_name):
            raise ValueError
    except:
        messagebox.showinfo(
            title='Error',
            message='Cluster/topic file not found! Please reselct the cluster file')
        static_topics_file_name = select_file('cluster')

    try:
        if contents[0][0:15] != 'results_folder,':
            raise ValueError
        result_folder_name = contents[0][15:len(contents[0]) - 1]
        if not os.path.isdir(result_folder_name):
            raise ValueError
    except:
        messagebox.showinfo(
            title='Error',
            message='Result folder not found! Please reselct the result folder')
        result_folder_name = select_file('folder')

    return result_folder_name, static_topics_file_name

# Save the current setting

def save_settings(settings_file_name, result_folder_name, static_topics_file_name):
    try:
        file = open(settings_file_name, 'w')
        line1 = 'results_folder,' + result_folder_name
        line2 = 'static_topics_file,' + static_topics_file_name
        file.write(line1)
        file.write('\n')
        file.write(line2)
        file.close()
    except:
        messagebox.showinfo(
            title='Error',
            message='Saving Settings Failed!')

# Helper function for reading cluster files

def read_file(filename):
    newlines = []
    try:
        f = open(filename)
    except:
        messagebox.showinfo(
            title='Error',
            message='Read ' + filename + " Failed!")
        return newlines
    c = f.readlines()
    for line in c:
        line = line.replace('\n', '')
        newlines.append(line)
    f.close()
    return newlines

# Loading the cluster (cluster stored as global variables)

def load_clusters(result_folder_name):
    clusterlines_file_name = result_folder_name + "\\novel_cluster_summary.txt"
    caselines_file_name = result_folder_name + "\\novel_caselines.txt"
    topiclines_file_name = result_folder_name + "\\novel_topicwords.txt"

    clusterlines = read_file(clusterlines_file_name)
    caselines = read_file(caselines_file_name)
    topiclines = read_file(topiclines_file_name)

    # Reading the cluster first
    for line in clusterlines:
        items = line.split('\t')
        if(len(items) != 7):
            continue
        c = Cluster(
            items[0],
            items[4],
            items[1],
            items[2],
            items[5],
            items[3],
            items[6],
            0,
            0
        )
        novel[c.index] = c

    # Reading the cases and assign them to their clusters
    for case in caselines:
        items = case.split('\t')
        if(len(items) != 9):
            continue
        c = Case(
            items[0],
            items[5],
            items[4],
            items[8],
            items[1],
            items[2],
            items[6],
            items[7],
            items[3],
            str(uuid.uuid4())
        )
        index = c.cluster_index
        if index in novel:
            novel[index].case_list.append(c)

    # Reading the words and assign them to their clusters
    for topic in topiclines:
        items = topic.split('\t')
        if(len(items) != 3):
            continue
        index = int(items[0])
        word = items[2]
        percent = items[1]
        if index in novel:
            novel[index].words_list.append([word, float(percent)])

    mon_clusterlines_file_name = result_folder_name + "\\monitored_cluster_summary.txt"
    mon_caselines_file_name = result_folder_name + "\\monitored_caselines.txt"
    mon_topiclines_file_name = result_folder_name + "\\monitored_topicwords.txt"

    # Only read the monitored files if they exist
    if (os.path.exists(mon_caselines_file_name) and
            os.path.exists(mon_clusterlines_file_name) and
            os.path.exists(mon_topiclines_file_name)):

        clusterlines = read_file(mon_clusterlines_file_name)
        caselines = read_file(mon_caselines_file_name)
        topiclines = read_file(mon_topiclines_file_name)

        for line in clusterlines:
            items = line.split('\t')
            if(len(items) != 7):
                continue
            c = Cluster(
                items[0],
                items[4],
                items[1],
                items[2],
                items[5],
                items[3],
                items[6],
                1,
                0
            )
            monitored[c.index] = c

        for case in caselines:
            items = case.split('\t')
            if(len(items) != 9):
                continue
            c = Case(
                items[0],
                items[5],
                items[4],
                items[8],
                items[1],
                items[2],
                items[6],
                items[7],
                items[3],
                str(uuid.uuid4())
            )
            index = c.cluster_index
            if index in monitored:
                monitored[index].case_list.append(c)

        for topic in topiclines:
            items = topic.split('\t')
            if(len(items) != 3):
                continue
            index = int(items[0])
            word = items[2]
            percent = items[1]
            if index in monitored:
                monitored[index].words_list.append([word, float(percent)])

# Helper function for sorting columns
# Sorted as string except for score

def treeview_sort_column(tv, col, reverse):
    l = [(tv.set(k, col), k) for k in tv.get_children('')]
    if col == 'score':
        l.sort(reverse=reverse, key=lambda x: float(x[0]))
    else:
        l.sort(reverse=reverse)

    # rearrange items in sorted positions
    for index, (val, k) in enumerate(l):
        tv.move(k, '', index)

    # reverse sort next time
    tv.heading(col, command=lambda: \
        treeview_sort_column(tv, col, not reverse))


# Updating the cluster details at the bottom when a new cluster/line is selected

def updating_case_table(event=None):
    cluster_id = table.selection()
    if len(table.selection()) == 0:
        cluster_id = list(novel.keys())
    cluster_id = int(cluster_id[0])
    cluster = novel[cluster_id]
    for chi in case_table.get_children():
        case_table.delete(chi)
    for case in cluster.case_list:
        case_table.insert(
            parent='',
            index='end',
            iid=case.uuid,
            values=case.to_tuple()
        )
    fig = Figure(figsize=(3, 3), dpi=100)
    plot1 = fig.add_subplot(111)
    words = np.array(cluster.words_list)
    weights = words[:, 1]
    weights = weights.astype(float)
    weights = weights / np.sum(weights)
    plot1.pie(weights, labels=words[:, 0])

    canvas = FigureCanvasTkAgg(fig, master=case_canvas)
    canvas.draw()

    canvas.get_tk_widget().grid(column=0, row=0)

# Helper function for opening a new window to add a cluster

def add_cluster():
    cluster_id = table.selection()
    if len(table.selection()) == 0:
        cluster_id = (0, list(novel.keys()))
    cluster_id = int(cluster_id[0])
    cluster = novel[cluster_id]
    add_cluster_helper(cluster)

# A new window that writes a cluster to static topic file
# New window only containing info of a single cluster

def add_cluster_helper(cluster):

    # Window layout
    new_window = Toplevel()
    new_window.config
    title = Label(new_window, text="Syndrome Summary")
    new_window.grid_rowconfigure(0, weight=1)
    new_window.grid_rowconfigure(1, weight=1)
    new_window.grid_rowconfigure(2, weight=3)
    new_window.grid_rowconfigure(3, weight=1)
    new_window.grid_rowconfigure(4, weight=1)
    new_window.grid_columnconfigure(0, weight=1)
    new_window.grid_columnconfigure(1, weight=1)
    new_window.configure(background='white')
    title.grid(row=0, column=0, columnspan=2, sticky='NESW')
    text = Label(
        new_window,
        text='Please review and make any necessary changes before incorporating this syndrome into your Semantic Scan.')
    text.grid(row=1, column=0, columnspan=2, sticky='NESW', )

    # Pie chart
    def update_graph():
        fig = Figure(figsize=(2, 3), dpi=100)
        plot1 = fig.add_subplot(111)
        words = np.array(cluster.words_list)
        weights = words[:, 1]
        weights = weights.astype(float)
        weights = weights / np.sum(weights)
        plot1.pie(weights, labels=words[:, 0])

        canvas = FigureCanvasTkAgg(fig, master=new_window)
        canvas.draw()

        canvas.get_tk_widget().grid(row=2, column=1, sticky='NESW')

    update_graph()

    # Editable word table
    words_table = ttk.Treeview(
        new_window
    )
    words_table.grid(column=0, row=2, sticky='NESW')

    words_table['columns'] = ('word', 'prob')
    words_table.column("#0", width=0, stretch=NO)
    words_table.column('word', anchor=CENTER, width=80)
    words_table.column('prob', anchor=CENTER, width=80)

    words_table.heading("#0", text="", anchor=CENTER)
    words_table.heading('word', text='Word', anchor=CENTER,
                        command=lambda: treeview_sort_column(words_table, "word", False))
    words_table.heading('prob', text='Probability', anchor=CENTER,
                        command=lambda: treeview_sort_column(words_table, "prob", False))

    for i, word in enumerate(cluster.words_list):
        words_table.insert(
            parent='',
            index='end',
            iid=word[0],
            values=word
        )

    # Function for editing words/probabilities
    def edit():
        word = words_table.selection()[0]
        new_new_window = Toplevel()
        e1 = Entry(new_new_window, width=50)
        e1.insert(0, word)
        e1.pack()
        e2 = Entry(new_new_window, width=50)
        prob = [a[1] for a in cluster.words_list if a[0] == word]
        prob = prob[0]
        e2.insert(0, prob)
        e2.pack()

        def edit2():
            new_word = e1.get()
            new_prob = e2.get()
            if not isfloat(new_prob):
                new_new_window.destroy()
                return
            nn = (new_word, new_prob)
            for i, a in enumerate(cluster.words_list):
                if a[0] == word:
                    cluster.words_list[i] = nn
                    break
            words_table.item(word, values=nn)
            update_graph()
            new_new_window.destroy()

        b = Button(new_new_window, text='Edit', command=edit2)
        b.pack()

    def delete():
        word = words_table.selection()[0]
        for i, a in enumerate(cluster.words_list):
            if a[0] == word:
                cluster.words_list.pop(i)
        update_graph()
        words_table.delete(word)

    edit_button = Button(new_window, text='Edit', command=edit)
    edit_button.grid(column=0, row=3)
    delete_button = Button(new_window, text='Delete', command=delete)
    delete_button.grid(column=1, row=3)

    # Whether monitor or not
    var = IntVar()
    include_checkbox = Checkbutton(
        new_window,
        text="Monitor this syndrome in future scans",
        variable=var,
        onvalue=1,
        offvalue=0)
    include_checkbox.deselect()
    include_checkbox.grid(column=0, row=4)

    def incorporate():
        try:
            f = open(static_topics_file_name, 'a')
        except:
            messagebox.showinfo(
                title='Error',
                message='Open ' + static_topics_file_name + ' failed!')
            new_window.destroy()
            return
        ret = ''
        ret = ret + str(int(var.get()))

        words = np.array(cluster.words_list)
        weights = words[:, 1]
        weights = weights.astype(float)
        weights = weights / np.sum(weights)
        words[:, 1] = weights

        for a in words:
            ret = ret + '_' + a[0] + '_' + str(a[1])
        ret = ret + '\n'
        f.write(ret)
        f.close()

        new_window.destroy()

    incorporate_button = Button(
        new_window,
        text='Incorporate Syndrome',
        command=incorporate)
    incorporate_button.grid(column=0, row=5, columnspan=2)

# TODO: Showing a help page
def help_program():
    pass


# TODO: Showing the about info
def about():
    pass

# Showing the time info about the topic and the cluster files
def fileinfo():
    new_window = Toplevel()
    new_window.geometry("300x200")
    last_modified = time.ctime(os.path.getmtime(result_folder_name + "\\novel_cluster_summary.txt"))
    time_label = Label(
        new_window,
        text=(
                "Results files: last modified at " +
                last_modified
        )
    )
    time_label.pack(expand=True, fill='both')

    last_modified2 = time.ctime(os.path.getmtime(static_topics_file_name))
    topic_time_label = Label(
        new_window,
        text=(
                "Static topic file: last modified at " +
                last_modified2
        )
    )
    topic_time_label.pack(expand=True, fill='both')


# Quit the program
def exit_program():
    root.destroy()


# A new window where monitored clusters are shown
# The main window only shows the novel ones
# Similar structures/functions
def show_monitored_clusters():
    def exit_this_window():
        root2.destroy()

    default_cluster = list(monitored.keys())

    def updating_case_table_monitored(event=None):
        cluster_id = table_monitored.selection()
        if len(table_monitored.selection()) == 0:
            cluster_id = default_cluster
        cluster_id = int(cluster_id[0])
        cluster = monitored[cluster_id]
        for chi in case_table_monitored.get_children():
            case_table_monitored.delete(chi)
        for case in cluster.case_list:
            case_table_monitored.insert(
                parent='',
                index='end',
                iid='-'+case.uuid,
                values=case.to_tuple()
            )
        fig = Figure(figsize=(3, 3), dpi=100)
        plot1 = fig.add_subplot(111)
        words = np.array(cluster.words_list)
        weights = words[:, 1]
        weights = weights.astype(float)
        weights = weights / np.sum(weights)
        plot1.pie(weights, labels=words[:, 0])

        canvas = FigureCanvasTkAgg(fig, master=case_canvas_monitored)
        canvas.draw()

        canvas.get_tk_widget().grid(column=0, row=0)

    def add_cluster_monitored():
        cluster_id = table_monitored.selection()
        if len(table_monitored.selection()) == 0:
            cluster_id = default_cluster
        cluster_id = int(cluster_id[0])
        cluster = monitored[cluster_id]
        add_cluster_helper(cluster)
    

    root2 = Toplevel()
    root2.state('zoomed')
    root2.grid_rowconfigure(0, weight=1)
    root2.grid_rowconfigure(1, weight=3)
    root2.grid_rowconfigure(2, weight=3)
    root2.grid_rowconfigure(3, weight=1)
    root2.grid_columnconfigure(0, weight=1)
    root2.grid_columnconfigure(1, weight=1)
    root2.grid_columnconfigure(2, weight=1)
    root2.configure(background='white')

    menubar = Menu(root2)

    filemenu = Menu(menubar, tearoff=0)
    #filemenu.add_command(label="Change Results Folder", command=change_results_folder)
    #filemenu.add_command(label="Change Static Topic File", command=change_static_topic)
    #filemenu.add_separator()
    #filemenu.add_command(label="About", command=about)
    #filemenu.add_separator()
    filemenu.add_command(label="Exit", command=exit_program)
    menubar.add_cascade(label='File', menu=filemenu)

    #helpmenu = Menu(menubar, tearoff=0)
    #helpmenu.add_command(label="Help", command=help_program)
    #helpmenu.add_command(label="Show cluster file info", command=fileinfo)
    #menubar.add_cascade(label="Help", menu=helpmenu)

    #optionmenu = Menu(menubar, tearoff=0)
    #optionmenu.add_command(label="Show statics", command=about)
    #menubar.add_cascade(label="Additional Options", menu=optionmenu)

    root2.config(menu=menubar)

    novel_button = Button(
        root2,
        text="Novel Clusters",
        borderwidth=5,
        relief='ridge',   
        command=exit_this_window
    )
    novel_button.grid(row=0, column=0, sticky='NESW')

    monitored_button = Button(
        root2,
        text="Monitored Clusters",
        borderwidth=5,
        relief='ridge',
        state=DISABLED)
    monitored_button.grid(row=0, column=2, sticky='NESW')

    cluster_frame = Frame(root2, borderwidth=5, relief='ridge')
    cluster_frame.grid(
        row=1,
        column=0,
        sticky='NESW',
        columnspan=3,
        rowspan=2
    )

    cluster_scroll = Scrollbar(cluster_frame)
    cluster_scroll.pack(side=RIGHT, fill=Y)

    table_label = Label(cluster_frame, text="Summary of Detected Monitored Clusters")
    table_label.pack()

    table_monitored = ttk.Treeview(
        cluster_frame,
        yscrollcommand=cluster_scroll.set)

    table_monitored.pack(expand=True, fill='both')

    cluster_scroll.config(command=table_monitored.yview)

    table_monitored['columns'] = ('score', 'startdate', 'enddate', 'words', 'location', 'age')

    table_monitored.column("#0", width=0, stretch=NO)
    table_monitored.column('score', anchor=CENTER, width=80)
    table_monitored.column('startdate', anchor=CENTER, width=40)
    table_monitored.column('enddate', anchor=CENTER, width=40)
    table_monitored.column('words', anchor=CENTER, width=160)
    table_monitored.column('location', anchor=CENTER, width=40)
    table_monitored.column('age', anchor=CENTER, width=80)

    table_monitored.heading("#0", text="", anchor=CENTER)
    table_monitored.heading('score', text='Score', anchor=CENTER,
                            command=lambda: treeview_sort_column(table_monitored, "score", False))
    table_monitored.heading('startdate', text='Start Date', anchor=CENTER,
                            command=lambda: treeview_sort_column(table_monitored, "startdate", False))
    table_monitored.heading('enddate', text='End Date', anchor=CENTER,
                            command=lambda: treeview_sort_column(table_monitored, "enddate", False))
    table_monitored.heading('words', text='Words in Learned Syndrome', anchor=CENTER,
                            command=lambda: treeview_sort_column(table_monitored, "words", False))
    table_monitored.heading('location', text='Affected Locations', anchor=CENTER,
                            command=lambda: treeview_sort_column(table_monitored, "location", False))
    table_monitored.heading('age', text='Affected Age Range', anchor=CENTER,
                            command=lambda: treeview_sort_column(table_monitored, "age", False))

    insert_rows(table_monitored, monitored)

    details_frame = Frame(root2, borderwidth=5, relief='ridge')
    details_frame.grid(
        row=3,
        column=0,
        sticky='NESW',
        columnspan=2,
        rowspan=1
    )

    details_label = Label(details_frame, text="Details for Selected Clusters")
    details_label.pack()

    case_table_monitored = ttk.Treeview(
        details_frame
    )

    case_table_monitored.pack(expand=True, fill='both')

    case_table_monitored['columns'] = ('date', 'time', 'location', 'words', 'icd', 'sex', 'age', 'id')

    case_table_monitored.column("#0", width=0, stretch=NO)
    case_table_monitored.column('date', anchor=CENTER, width=40)
    case_table_monitored.column('time', anchor=CENTER, width=40)
    case_table_monitored.column('location', anchor=CENTER, width=40)
    case_table_monitored.column('words', anchor=CENTER, width=80)
    case_table_monitored.column('icd', anchor=CENTER, width=40)
    case_table_monitored.column('sex', anchor=CENTER, width=40)
    case_table_monitored.column('age', anchor=CENTER, width=40)
    case_table_monitored.column('id', anchor=CENTER, width=80)

    case_table_monitored.heading("#0", text="", anchor=CENTER)
    case_table_monitored.heading('date', text='Date', anchor=CENTER,
                                 command=lambda: treeview_sort_column(case_table_monitored, "date", False))
    case_table_monitored.heading('time', text='Time', anchor=CENTER,
                                 command=lambda: treeview_sort_column(case_table_monitored, "time", False))
    case_table_monitored.heading('location', text='Location', anchor=CENTER,
                                 command=lambda: treeview_sort_column(case_table_monitored, "location", False))
    case_table_monitored.heading('words', text='Chief Complaint', anchor=CENTER,
                                 command=lambda: treeview_sort_column(case_table_monitored, "words", False))
    case_table_monitored.heading('icd', text='ICD', anchor=CENTER,
                                 command=lambda: treeview_sort_column(case_table_monitored, "icd", False))
    case_table_monitored.heading('sex', text='Sex', anchor=CENTER,
                                 command=lambda: treeview_sort_column(case_table_monitored, "sex", False))
    case_table_monitored.heading('age', text='Age Group', anchor=CENTER,
                                 command=lambda: treeview_sort_column(case_table_monitored, "age", False))
    case_table_monitored.heading('id', text='VISIT ID', anchor=CENTER,
                                 command=lambda: treeview_sort_column(case_table_monitored, "id", False))

    canvas_frame = Frame(root2, borderwidth=5, relief='ridge')
    canvas_frame.grid(
        row=3,
        column=2,
        sticky='NESW',
        rowspan=1,
        columnspan=1
    )

    canvas_label = Label(canvas_frame, text="Pie Chart for Selected Clusters")
    canvas_label.pack()

    case_canvas_monitored = Frame(canvas_frame)

    case_canvas_monitored.pack()

    monitor_cluster_button = Button(
        canvas_frame, 
        text="Include Cluster in Future Runs", 
        command=add_cluster_monitored,
        state = DISABLED if len(monitored) == 0 else NORMAL
    )

    monitor_cluster_button.pack(expand=True, fill='x')

    updating_case_table_monitored()

    table_monitored.bind("<<TreeviewSelect>>", updating_case_table_monitored)

def main(args):
    # global info
    global result_folder_name
    global static_topics_file_name
    global novel
    global monitored
    global table
    global case_table
    global case_canvas
    global root

    # Setting up the main window
    root = Tk()
    root.title("Disease Surveillance Visualization")
    root.state('zoomed')
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=3)
    root.grid_rowconfigure(2, weight=3)
    root.grid_rowconfigure(3, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)
    root.configure(background='white')

    # Setting up the menu
    menubar = Menu(root)

    filemenu = Menu(menubar, tearoff=0)
    #filemenu.add_command(label="Change Results Folder", command=change_results_folder)
    #filemenu.add_command(label="Change Static Topic File", command=change_static_topic)
    #filemenu.add_separator()
    #filemenu.add_command(label="About", command=about)
    #filemenu.add_separator()
    filemenu.add_command(label="Exit", command=exit_program)
    menubar.add_cascade(label='File', menu=filemenu)

    #helpmenu = Menu(menubar, tearoff=0)
    #helpmenu.add_command(label="Help", command=help_program)
    #helpmenu.add_command(label="Show cluster file info", command=fileinfo)
    #menubar.add_cascade(label="Help", menu=helpmenu)

    #optionmenu = Menu(menubar, tearoff=0)
    #optionmenu.add_command(label="Show statics", command=about)
    #menubar.add_cascade(label="Additonal Options", menu=optionmenu)

    root.config(menu=menubar)

    # Setting up the main window frame
    loading_frame = Frame(root,
                      borderwidth=5,
                      relief="ridge")
    loading_frame.grid(
        row=0,
        column=0,
        rowspan=4,
        columnspan=3
    )

    loading_label = Label(loading_frame, text='Loading...')
    loading_label.pack(expand=True, fill='both')

    result_folder_name = args.results_folder
    static_topics_file_name = args.monitored_topic_file

    no_file_flag = False
    # Program exiting if no files found
    if result_folder_name == '':
        messagebox.showinfo(
            title='Error',
            message='No file found! Program Exiting')
        no_file_flag = True
    
    elif static_topics_file_name == '':
        messagebox.showinfo(
            title='Error',
            message='No file found! Program Exiting')
        no_file_flag = True

    novel = {}
    monitored = {}

    # Read and load the clusters onto the main window/frame
    if(no_file_flag):
        root.destroy()
        exit()
    else:
        load_clusters(result_folder_name)

    # Finishing Loading
    loading_label.destroy()

    # Buttons for switching between novel clusters and monitored clusters
    novel_button = Button(
        root,
        text="Novel Clusters",
        borderwidth=5,
        relief='ridge',
        state=DISABLED)
    novel_button.grid(row=0, column=0, sticky='NESW')

    monitored_button = Button(
        root,
        text="Monitored Clusters",
        borderwidth=5,
        relief='ridge',
        command=show_monitored_clusters)
    monitored_button.grid(row=0, column=2, sticky='NESW')

    # A frame to store the cluster table
    cluster_frame = Frame(root, borderwidth=5, relief='ridge')
    cluster_frame.grid(
        row=1,
        column=0,
        sticky='NESW',
        columnspan=3,
        rowspan=2
    )

    cluster_scroll = Scrollbar(cluster_frame)
    cluster_scroll.pack(side=RIGHT, fill=Y)

    table_label = Label(cluster_frame, text="Summary of Detected Novel Clusters")
    table_label.pack()

    # Setting up the cluster table
    table = ttk.Treeview(
        cluster_frame,
        yscrollcommand=cluster_scroll.set)

    table.pack(expand=True, fill='both')

    cluster_scroll.config(command=table.yview)

    table['columns'] = ('score', 'startdate', 'enddate', 'words', 'location', 'age')

    table.column("#0", width=0, stretch=NO)
    table.column('score', anchor=CENTER, width=80)
    table.column('startdate', anchor=CENTER, width=40)
    table.column('enddate', anchor=CENTER, width=40)
    table.column('words', anchor=CENTER, width=160)
    table.column('location', anchor=CENTER, width=40)
    table.column('age', anchor=CENTER, width=80)

    table.heading("#0", text="", anchor=CENTER)
    table.heading('score', text='Score', anchor=CENTER,
              command=lambda: treeview_sort_column(table, "score", False))
    table.heading('startdate', text='Start Date', anchor=CENTER,
              command=lambda: treeview_sort_column(table, "startdate", False))
    table.heading('enddate', text='End Date', anchor=CENTER,
              command=lambda: treeview_sort_column(table, "enddate", False))
    table.heading('words', text='Words in Learned Syndrome', anchor=CENTER,
              command=lambda: treeview_sort_column(table, "words", False))
    table.heading('location', text='Affected Locations', anchor=CENTER,
              command=lambda: treeview_sort_column(table, "location", False))
    table.heading('age', text='Affected Age Range', anchor=CENTER,
              command=lambda: treeview_sort_column(table, "age", False))

    # Inserting the clusters rows to the table
    insert_rows(table, novel)

    # Setting up the lower frame for the cluster detail
    details_frame = Frame(root, borderwidth=5, relief='ridge')
    details_frame.grid(
        row=3,
        column=0,
        sticky='NESW',
        columnspan=2,
        rowspan=1
   )

    details_label = Label(details_frame, text="Details for Selected Clusters")
    details_label.pack()

    # A table for cases in a specfic cluster
    case_table = ttk.Treeview(
        details_frame
    )

    case_table.pack(expand=True, fill='both')

    case_table['columns'] = ('date', 'time', 'location', 'words', 'icd', 'sex', 'age', 'id')

    case_table.column("#0", width=0, stretch=NO)
    case_table.column('date', anchor=CENTER, width=40)
    case_table.column('time', anchor=CENTER, width=40)
    case_table.column('location', anchor=CENTER, width=40)
    case_table.column('words', anchor=CENTER, width=80)
    case_table.column('icd', anchor=CENTER, width=40)
    case_table.column('sex', anchor=CENTER, width=40)
    case_table.column('age', anchor=CENTER, width=40)
    case_table.column('id', anchor=CENTER, width=80)

    case_table.heading("#0", text="", anchor=CENTER)
    case_table.heading('date', text='Date', anchor=CENTER,
                   command=lambda: treeview_sort_column(case_table, "date", False))
    case_table.heading('time', text='Time', anchor=CENTER,
                   command=lambda: treeview_sort_column(case_table, "time", False))
    case_table.heading('location', text='Location', anchor=CENTER,
                   command=lambda: treeview_sort_column(case_table, "location", False))
    case_table.heading('words', text='Chief Complaint', anchor=CENTER,
                   command=lambda: treeview_sort_column(case_table, "words", False))
    case_table.heading('icd', text='ICD', anchor=CENTER,
                   command=lambda: treeview_sort_column(case_table, "icd", False))
    case_table.heading('sex', text='Sex', anchor=CENTER,
                   command=lambda: treeview_sort_column(case_table, "sex", False))
    case_table.heading('age', text='Age Group', anchor=CENTER,
                   command=lambda: treeview_sort_column(case_table, "age", False))
    case_table.heading('id', text='VISIT ID', anchor=CENTER,
                   command=lambda: treeview_sort_column(case_table, "id", False))

    # Setting up the pie chart section
    canvas_frame = Frame(root, borderwidth=5, relief='ridge')
    canvas_frame.grid(
        row=3,
        column=2,
        sticky='NESW',
        rowspan=1,
        columnspan=1
    )

    canvas_label = Label(canvas_frame, text="Pie Chart for Selected Clusters")
    canvas_label.pack()

    case_canvas = Frame(canvas_frame)

    case_canvas.pack()

    monitor_cluster_button = Button(canvas_frame, text="Include Cluster in Future Runs", command=add_cluster)

    monitor_cluster_button.pack(expand=True, fill='x')

    updating_case_table()

    table.bind("<<TreeviewSelect>>", updating_case_table)

    root.mainloop()

if __name__ == '__main__':
    main(FLAGS)