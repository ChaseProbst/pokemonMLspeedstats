import json
import sys
import scipy
import numpy
import matplotlib
import sklearn

from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# Notes on formats
# I think that stats is in the form { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }
# b/c of this file https://github.com/pkmn/stats/blob/master/stats/src/parser.ts

natures = { 'Adamant':{ '+':'atk', '-':'spa'}, 'Bold':{ '+':'def', '-':'atk'}, 'Brave':{ '+':'atk', '-':'spe'}, 'Calm':{ '+':'spd', '-':'atk'}, 'Careful':{ '+':'spd', '-':'spa'}, 'Gentle':{ '+':'spd', '-':'def'}, 'Hasty':{ '+':'spe', '-':'def'}, 'Impish':{ '+':'def', '-':'spa'}, 'Joly':{ '+':'spe', '-':'spa'}, 'Lax':{ '+':'def', '-':'spa'}, 'Lonely':{ '+':'atk', '-':'def'}, 'Mild':{ '+':'spa', '-':'def'}, 'Modest':{ '+':'spa', '-':'atk'}, 'Naive':{ '+':'spe', '-':'spd'}, 'Naughty':{ '+':'atk', '-':'spd'}, 'Quiet':{ '+':'spa', '-':'spe'}, 'Rash':{ '+':'spa', '-':'spd'}, 'Relaxed':{ '+':'def', '-':'spe'}, 'Sassy':{ '+':'spd', '-':'spe'}, 'Timid':{ '+':'spe', '-':'atk'}}
#print(natures)
def calculate_speed(pokemon_dict, speed_ev, nature):
    base_speed = pokemondict['base']['Speed']
    mult = 1.0
    if natures[nature]['+'] == 'spe':
        mult = 1.1
    if natures[nature]['-'] == 'spe':
        mult = .9
    speed = math.floor( ( (2*base_speed + 31 + math.floor(speedev/4))/2 +5) * mult )
    return speed

def calculate_ev(speed_in, pokemon_dict, nature):
    base_speed = pokemondict['base']['Speed']
    mult = 1.0
    if natures[nature]['+'] == 'spe':
        mult = 1.1
    if natures[nature]['-'] == 'spe':
        mult = .9
    ev_out = ( (( math.ceil(speedin/mult) -5) * 2) - (2*base_speed +31) )*4
    #speed = math.floor( ( (2*base_speed + 31 + math.floor(speedev/4))/2 +5) * mult )
    #speed = (2*base_speed + 31 + math.floor(speedev/4))/2 * mult
    return ev_out

if __name__ == '__main__':
    json_dir = '../Json'
    name_prefix = '/gen8battlestadiumdoubles-'
    name_suffixes = ['0','1500','1630','1760']
    f = open(json_dir+name_prefix+name_suffixes[2]+'.json')
    fp = open(json_dir+'/pokedex.json')

    data = json.load(f)
    pokedex = json.load(fp)
    #print(data)
    #print('something')

    pokemon = data['data']


    viable = {}
    #print(pokemon)
    for x in pokemon:
        spreads = pokemon[x]['Spreads'].keys()
        categories = pokemon[x].keys()
        # get&store total item count
        item_count = 0
        for c in pokemon[x]['Items'].values():
            #print(c)
            item_count += c
        pokemon[x]['Items']['total count'] = item_count

        #### look at speed items
        # if 'quickclaw' in pokemon[x]['Items']:
        #     print(x, pokemon[x]['Items']['quickclaw'])
        # if 'salacberry' in pokemon[x]['Items']:
        #     print(x, pokemon[x]['Items']['salacberry'])
        # if 'choicescarf' in pokemon[x]['Items']:
        #     print(x, pokemon[x]['Items']['choicescarf'])
        
        #print(x, categories)
        #print(x,pokemon[x]['Spreads'])
    #for y in pokemon['Dracovish']['Spreads']:
    #print( sorted(pokemon['Dracovish']['Spreads']) )

    for x in pokedex:
        
        #print(x['id'],x['name']['english'], x['type'], x['base'])
        if x['name']['english'] in pokemon:
            entry = {}
            entry['id'] = x['id']
            entry['type'] = x['type']
            entry['base'] = x['base']

            # ev spread and nature
            stat_choices = []
            for k in pokemon[x['name']['english']]['Spreads']:
                nature, spread = k.split(':',1)
                #print(nature, spread, pokemon[x['name']['english']]['Spreads'][k])
                frq_ = pokemon[x['name']['english']]['Spreads'][k]
                hp_, atk_, def_, spa_, spd_, spe_ = spread.split('/')
                #print( hp_, atk_, def_, spa_, spd_, spe_ ) 
                stat_choice = {'frq':frq_, 'nature':nature, 'hp':hp_, 'atk':atk_, 'def':def_, 'spa':spa_, 'spd':spd_, 'spe':spe_}
                #print(stat_choice)
                stat_choices.append(stat_choice)
            entry['stat choices'] = stat_choices

            # speed items
            Items_ = {}
            # quick claw % 
            if 'quickclaw' in pokemon[x['name']['english']]['Items']:
                #print( pokemon[x['name']['english']]['Items']['quickclaw'],pokemon[x['name']['english']]['Items']['total count'])
                Items_['quickclaw'] = pokemon[x['name']['english']]['Items']['quickclaw'] / pokemon[x['name']['english']]['Items']['total count']
            else:
                Items_['quickclaw'] = 0
            # salacberry % 
            if 'salacberry' in pokemon[x['name']['english']]['Items']:
                #print( pokemon[x['name']['english']]['Items']['salacberry'],pokemon[x['name']['english']]['Items']['total count'])
                Items_['salacberry'] = pokemon[x['name']['english']]['Items']['salacberry'] / pokemon[x['name']['english']]['Items']['total count']
            else:
                Items_['salacberry'] = 0
            # choice scarf % 
            if 'choicescarf' in pokemon[x['name']['english']]['Items']:
                #print( pokemon[x['name']['english']]['Items']['choicescarf'],pokemon[x['name']['english']]['Items']['total count'])
                Items_['choicescarf'] = pokemon[x['name']['english']]['Items']['choicescarf'] / pokemon[x['name']['english']]['Items']['total count']
            else:
                Items_['choicescarf'] = 0
            entry['Items'] = Items_

            viable[ x['name']['english'] ] = entry

            #print( x['id'], x['name']['english'])#, pokemon[x['name']['english']] )
        #print(x)
    
    #print( viable )


#----------Code to run the ML algorithms------------


#print(viable['dracovish']['id'],viable['dracovish']['type'],viable['dracovish']['base'],viable['dracovish']['Items'])
X = []
y = []
pokemonNames = []
print("Please input the name of the pokemon")
pokemonNames.append(input())
#pokemonNames.append(input())


for k in pokemonNames:
    for x in viable[k]['stat choices']:
        """ if x['nature'] in natures:
            float(x[natures[x['nature']]['+']]) *= 1.1
            x[natures[x['nature']]['-']] *= 0.9 """
        dataSet = list(x.values())
        data = [int(i) for i in dataSet[2:]]
        data.insert(0,int(dataSet[0]))
              
        
        #y.append(data[6])
        """ X.append(data[1:6])    
        y.append(data[6]) """

        for i in range(0,data[0]):
            X.append(data[1:6])
            if data[6] is 252:
                y.append(3)
            elif data[6] >= 204:
                y.append(2)
            elif data[6] >= 148:
                y.append(1)
            else:
                y.append(0)
        


    #print(X)

    #print(y)

    X_train, X_validation, Y_train, Y_validation = train_test_split(X,y,test_size=0.2,random_state=1,shuffle=True)

    #print(X_train, Y_validation)


    # Creating list of models
    models = []
    models.append(('Logistic', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('DTree', DecisionTreeClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    #models.append(('NB', GaussianNB()))
    #models.append(('MinikMeans', MiniBatchKMeans()))
    #models.append(('kMeans', KMeans()))
    models.append(('SVM', SVC(gamma='auto')))

    #running the different models and 
    results=[]
    names=[]
    #predictions=[]
    for name, model in models:
        kfold = StratifiedKFold(n_splits=5 ,shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    pyplot.boxplot(results, labels=names)
    pyplot.title('Algorithm Accuracy')

    for name, model in models:
        model.fit(X_train, Y_train)
        #predictThis = [ [ int(viable[k]['stat choices'][0]['hp']), int(viable[k]['stat choices'][0]['atk']), int(viable[k]['stat choices'][0]['def']), int(viable[k]['stat choices'][0]['spa']), int(viable[k]['stat choices'][0]['spd'])] ]
        #predictions = model.predict(predictThis)
        predictions = model.predict(X_validation)
        print('Model: ', name)
        print('Predictions: ', predictions)
        #correctSPE = int(viable[k]['stat choices'][0]['spe'])
        #code to check the accuracy of prediction
        """ if correctSPE is 252:
            correctSPE = 3
        elif correctSPE >= 204:
            correctSPE = 2
        elif correctSPE >= 148:
            correctSPE = 1
        else:
            correctSPE = 0  
        print('Correct value: ', correctSPE)"""
        print(accuracy_score(Y_validation, predictions))
        print(confusion_matrix(Y_validation, predictions))
        print(classification_report(Y_validation, predictions))
    

pyplot.show()
