import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from time import time
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import Pipeline
import warnings
import traceback
import pickle
warnings.simplefilter("ignore")




# start of prediction 
# take the two team input
# get details of those two team
# and calculate the odds of winning

start = time()
## Fetching data
#Connecting to database
t1=""
def getprediction(team1,team2):
    global t1
    t1 = team1
    final_label =["Win","Defeat","Draw"]
    database =  '/home/ruvel/Documents/footbal lscore/database.sqlite'
    conn = sqlite3.connect(database)
    bid =2
    #Defining the number of jobs to be run in parallel during grid search
    n_jobs = 1 #Insert number of parallel jobs here
    #Fetching required data tables
    nom1= len(team1)
    player_data = pd.read_sql("SELECT * FROM Player;", conn)
    player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
    team_data = pd.read_sql("SELECT * FROM Team;", conn)
    team_list =[]
    for t in team_data["team_long_name"]:
        team_list.append(t)
        print(t)
    print(team_list)
    nom2 = len(team2)
    match_data = pd.read_sql("SELECT * FROM Match;", conn)
    #print(team_data[team_data['team_long_name'].str.contains(team1)])
    print(team_data[team_data['team_long_name'] == team1])
    if len(team_data[team_data['team_long_name'] == team1])==0:
        return "Error , no team name "+str(team1)
    print(team_data[team_data['team_long_name'] == team2])
    if len(team_data[team_data['team_long_name'] == team2])==0:
        return "Error , no team name "+str(team2)
    if nom2>nom1:
       bid = 1

    #Reduce match data to fulfill run time requirements
    rows = ['home_team_goals_difference', 
            'away_team_goals_difference', 
            'games_won_home_team', 
            'games_won_away_team', 
            'games_against_won', 
            'games_against_lost', 
            'home_player_1_overall_rating', 
            'home_player_2_overall_rating',
            'home_player_3_overall_rating', 
            'home_player_4_overall_rating', 
            'home_player_5_overall_rating', 
            'home_player_6_overall_rating',
            'home_player_7_overall_rating',
            'home_player_8_overall_rating', 
            'home_player_9_overall_rating', 
            'home_player_10_overall_rating', 
            'home_player_11_overall_rating',
            'away_player_1_overall_rating', 
            'away_player_2_overall_rating', 
            'away_player_3_overall_rating', 
            'away_player_4_overall_rating', 
            'away_player_5_overall_rating', 
            'away_player_6_overall_rating', 
            'away_player_7_overall_rating', 
            'away_player_8_overall_rating', 
            'away_player_9_overall_rating', 
            'away_player_10_overall_rating', 
            'away_player_11_overall_rating', 
            "season",
            "stage"
            ]
    print(rows)
    if nom1>nom2:
       bid = 0
    start = time()
    player_data = pd.read_sql("SELECT * FROM Player;", conn)
    player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
    filename = "GaussianNB.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    try:
        nom = loaded_model.score(X_test, Y_test)
    except:
        return convert_odds_to_prob(None,bid,final_label)

    match_data.dropna(subset = rows, inplace = True)
    match_data = match_data.tail(1500)

def predict_labels(clf, best_pipe, features, target):
    ''' Makes predictions using a fit classifier based on scorer. '''
    
    #Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(best_pipe.named_steps['dm_reduce'].transform(features))
    end = time()
    
    #print and return results
    print("Made predictions in {:.4f} seconds".format(end - start))
    return accuracy_score(target.values, y_pred)
    
def train_calibrate_predict(clf, dm_reduction, X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, cv_sets, params, scorer, jobs, 
                            use_grid_search = True, **kwargs):
    ''' Train and predict using a classifer based on scorer. '''
    
    #Indicate the classifier and the training set size
    print("Training a {} with {}...".format(clf.__class__.__name__, dm_reduction.__class__.__name__))
    
    #Train the classifier
    best_pipe = train_classifier(clf, dm_reduction, X_train, y_train, cv_sets, params, scorer, jobs)
    
    #Calibrate classifier
    print("Calibrating probabilities of classifier...")
    start = time()    
    clf = CalibratedClassifierCV(best_pipe.named_steps['clf'], cv= 'prefit', method='isotonic')
    clf.fit(best_pipe.named_steps['dm_reduce'].transform(X_calibrate), y_calibrate)
    end = time()
    print("Calibrated {} in {:.1f} minutes".format(clf.__class__.__name__, (end - start)/60))
    
    # print the results of prediction for both training and testing
    print("Score of {} for training set: {:.4f}.".format(clf.__class__.__name__, predict_labels(clf, best_pipe, X_train, y_train)))
    print("Score of {} for test set: {:.4f}.".format(clf.__class__.__name__, predict_labels(clf, best_pipe, X_test, y_test)))
    
    #Return classifier, dm reduction, and label predictions for train and test set
    return clf, best_pipe.named_steps['dm_reduce'], predict_labels(clf, best_pipe, X_train, y_train), predict_labels(clf, best_pipe, X_test, y_test)
        
def convert_odds_to_prob(match_odds,bid,final_label):
    ''' Converts bookkeeper odds to probabilities. '''
    return t1 +" "+ final_label[bid]
    #Define variables
    match_id = match_odds.loc[:,'match_api_id']
    bookkeeper = match_odds.loc[:,'bookkeeper']    
    win_odd = match_odds.loc[:,'Win']
    draw_odd = match_odds.loc[:,'Draw']
    loss_odd = match_odds.loc[:,'Defeat']
    
    #Converts odds to prob
    win_prob = 1 / win_odd
    draw_prob = 1 / draw_odd
    loss_prob = 1 / loss_odd
    
    total_prob = win_prob + draw_prob + loss_prob
    
    probs = pd.DataFrame()
    
    #Define output format and scale probs by sum over all probs
    probs.loc[:,'match_api_id'] = match_id
    probs.loc[:,'bookkeeper'] = bookkeeper
    probs.loc[:,'Win'] = win_prob / total_prob
    probs.loc[:,'Draw'] = draw_prob / total_prob
    probs.loc[:,'Defeat'] = loss_prob / total_prob
    
    #Return probs and meta data
    return probs
    
def get_bookkeeper_data(matches, bookkeepers, horizontal = True):
    ''' Aggregates bookkeeper data for all matches and bookkeepers. '''
    
    bk_data = pd.DataFrame()
    
    #Loop through bookkeepers
    for bookkeeper in bookkeepers:

        #Find columns containing data of bookkeeper
        temp_data = matches.loc[:,(matches.columns.str.contains(bookkeeper))]
        temp_data.loc[:, 'bookkeeper'] = str(bookkeeper)
        temp_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']
        
        #Rename odds columns and convert to numeric
        cols = temp_data.columns.values
        cols[:3] = ['Win','Draw','Defeat']
        temp_data.columns = cols
        temp_data.loc[:,'Win'] = pd.to_numeric(temp_data['Win'])
        temp_data.loc[:,'Draw'] = pd.to_numeric(temp_data['Draw'])
        temp_data.loc[:,'Defeat'] = pd.to_numeric(temp_data['Defeat'])
        
        #Check if data should be aggregated horizontally
        if(horizontal == True):
            
            #Convert data to probs
            temp_data = convert_odds_to_prob(temp_data)
            temp_data.drop('match_api_id', axis = 1, inplace = True)
            temp_data.drop('bookkeeper', axis = 1, inplace = True)
            
            #Rename columns with bookkeeper names
            win_name = bookkeeper + "_" + "Win"
            draw_name = bookkeeper + "_" + "Draw"
            defeat_name = bookkeeper + "_" + "Defeat"
            temp_data.columns.values[:3] = [win_name, draw_name, defeat_name]

            #Aggregate data
            bk_data = pd.concat([bk_data, temp_data], axis = 1)
        else:
            #Aggregate vertically
            bk_data = bk_data.append(temp_data, ignore_index = True)
    
    #If horizontal add match api id to data
    if(horizontal == True):
        temp_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']
    
    #Return bookkeeper data
    return bk_data
    
def get_bookkeeper_probs(matches, bookkeepers, horizontal = False):
    ''' Get bookkeeper data and convert to probabilities for vertical aggregation. '''
    
    #Get bookkeeper data
    data = get_bookkeeper_data(matches, bookkeepers, horizontal = False)
    
    #Convert odds to probabilities
    probs = convert_odds_to_prob(data)
    
    #Return data
    return probs

def plot_confusion_matrix(y_test, X_test, clf, dim_reduce, path, cmap=plt.cm.Blues, normalize = False):    
    ''' Plot confusion matrix for given classifier and data. '''
    
    #Define label names and get confusion matrix values
    labels = ["Win", "Draw", "Defeat"]
    cm = confusion_matrix(y_test, clf.predict(dim_reduce.transform(X_test)), labels)
    
    #Check if matrix should be normalized
    if normalize == True:
        
        #Normalize
        cm = cm.astype('float') / cm.sum()
        
    #Configure figure
    sns.set_style("whitegrid", {"axes.grid" : False})
    fig = plt.figure(1)    
    plt.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
    title= "Confusion matrix of a {} with {}".format(best_clf.base_estimator.__class__.__name__, best_dm_reduce.__class__.__name__)   
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.show()
    
# team_1_name = input("Enter Team one full name: ")
# print("Your first team name is "+team_1_name)
# team_2_name = input("Enter Team two full name: ")
# print("Your second team name is "+team_2_name)
# print("Working on prediction.........")
# print(getprediction(team_1_name,team_2_name))
