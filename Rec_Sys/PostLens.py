import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict

class postLens:

    postID_to_name = {}
    name_to_postID = {}
    ratingsPath = './ratings.csv'
    postsPath = './posts.csv'
    
    def loadpostLensLatestSmall(self):

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        ratingsDataset = 0
        self.postID_to_name = {}
        self.name_to_postID = {}

        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)
        # ratingsDataset.raw_ratings = [r for r in ratingsDataset.raw_ratings if r[2] >= 4.0]

        with open(self.postsPath, newline='', encoding='ISO-8859-1') as csvfile:
                postReader = csv.reader(csvfile)
                next(postReader)  #Skip header line
                for row in postReader:
                    postID = int(row[0])
                    postName = row[1]
                    self.postID_to_name[postID] = postName
                    self.name_to_postID[postName] = postID

        return ratingsDataset

    def getUserRatings(self, user):
        userRatings = []
        hitUser = False
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                userID = int(row[0])
                if (user == userID):
                    postID = int(row[1])
                    rating = float(row[2])
                    userRatings.append((postID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break

        return userRatings

    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                postID = int(row[1])
                ratings[postID] += 1
        rank = 1
        for postID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[postID] = rank
            rank += 1
        return rankings
    
    def getGenres(self):
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0
        with open(self.postsPath, newline='', encoding='ISO-8859-1') as csvfile:
            postReader = csv.reader(csvfile)
            next(postReader)  #Skip header line
            for row in postReader:
                postID = int(row[0])
                genreList = row[2].split('|')
                genreIDList = []
                for genre in genreList:
                    if genre in genreIDs:
                        genreID = genreIDs[genre]
                    else:
                        genreID = maxGenreID
                        genreIDs[genre] = genreID
                        maxGenreID += 1
                    genreIDList.append(genreID)
                genres[postID] = genreIDList
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (postID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[postID] = bitfield            
        
        return genres
    
    def getYears(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.postsPath, newline='', encoding='ISO-8859-1') as csvfile:
            postReader = csv.reader(csvfile)
            next(postReader)
            for row in postReader:
                postID = int(row[0])
                title = row[1]
                m = p.search(title)
                year = m.group(1)
                if year:
                    years[postID] = int(year)
        return years
    
    def getMiseEnScene(self):
        mes = defaultdict(list)
        with open("LLVisualFeatures13K_Log.csv", newline='') as csvfile:
            mesReader = csv.reader(csvfile)
            next(mesReader)
            for row in mesReader:
                postID = int(row[0])
                avgShotLength = float(row[1])
                meanColorVariance = float(row[2])
                stddevColorVariance = float(row[3])
                meanMotion = float(row[4])
                stddevMotion = float(row[5])
                meanLightingKey = float(row[6])
                numShots = float(row[7])
                mes[postID] = [avgShotLength, meanColorVariance, stddevColorVariance,
                   meanMotion, stddevMotion, meanLightingKey, numShots]
        return mes
    
    def getpostName(self, postID):
        if postID in self.postID_to_name:
            return self.postID_to_name[postID]
        else:
            return ""
        
    def getpostID(self, postName):
        if postName in self.name_to_postID:
            return self.name_to_postID[postName]
        else:
            return 0