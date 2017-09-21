# Houda Alberts
from __future__ import division
import os
import networkx as nx
from networkx.algorithms.components.connected import connected_components
import community
import glob
import json
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math
import sys
from collections import Counter

def getInfoData(firstPath):
    """ Given a path to a directory with an info.json file, return its content
        with information. """
    infoFile = glob.glob(firstPath + "info.json")[0]
    with open(infoFile) as data_file:
        data = json.load(data_file)
    return data

def createDictionaryData(firstPath, allProblems):
    """ Given a path to a directory with problem sets containing .txt documents
        and a list with the name of each folder that corresponds to a problem,
        return a dictionary with the architecture of the given path where
        the value is the text within a file lowercased and stripped. """
    newDataDict = {}
    for currentProblem in allProblems:
        allSubProblems = glob.glob(firstPath + currentProblem + "/*")
        newData = {}
        for subProblem in allSubProblems:
            probl = subProblem.split("/")[-1]
            with open(subProblem) as curData:
                problem = unicode(curData.read(), "utf-8")
                text = problem.strip().lower()
                allWords = text.split()
                count = Counter(allWords)
                newT = []
                for term in allWords:
                    value = count[term]
                    if value != 1:
                       newT.append(term)
                    else:
                       newT.append("*"*len(term))
                newData[probl] = (" ").join(newT)
                #newData[probl] = text
        newDataDict[currentProblem] = newData
    return newDataDict

def extractData(firstPath):
    """ Given a path to a directory with problem sets containing .txt documents,
        create a dictionary with the architecture of the given folder, where the
        value is the text within a file lowercased and stripped. """
    data = getInfoData(firstPath)
    allProblems = [problem["folder"] for problem in data]
    newDataDict = createDictionaryData(firstPath, allProblems)
    return newDataDict

def createFeaturesNGrams(text, n, k, mode, opt):
    """ Given a piece of text, the value of n of the n-gram, the amount of
        features that will be used (k), whether unique or top is chosen and
        an option for words or characters as n-grams, return the requested
        n-grams. """
    if opt == "char":
        allNgrams = [text[i:i+n] for i in range(len(text)-n+1)]
    elif opt == "word":
        inputList = text.split()
        allNgrams = zip(*[inputList[i:] for i in range(n)])
    if mode == "normal":
        return allNgrams
    if mode == "top":
        countObj = Counter(allNgrams).most_common(k)
        stayingFeats = [c[0] for c in countObj]
        return [ngram for ngram in allNgrams if ngram in stayingFeats]
    countObj = Counter(allNgrams)
    newNs = [ngram for ngram in allNgrams
                   if countObj[ngram] != 1]
    if mode == "remove":
        if newFours != []:
            return newNs
        else:
            return allNgrams
    if mode == "one":
        uniques = [ngram for ngram in allNgrams
                   if countObj[ngram] == 1]
        return newNs + ["unique"]*len(uniques)

def getSpatiumDistances(newDataDict, option, k, n, modeFeats):
    """ Given a dictionary containing the text for each document within each
        problem set, an option for unique terms/top features, an amount for the
        top k features, the value n for the character n-grams and whether it is
        character or words n-grams with the given n, return the same
        structured dictionary with spatium distances, the averages and standard
        deviations.
        """
    allProbl = newDataDict.keys()
    allInfo = {}
    cv = CountVectorizer(encoding="unicode", analyzer=lambda text:
                         createFeaturesNGrams(text, n, k, option, modeFeats))
    for currentProbl in allProbl:
        currentProblem = newDataDict[currentProbl]
        allDocs = currentProblem.keys()
        allInformation = {}
        for doc1 in allDocs:
            docDict = {}
            avg = 0
            textDoc1 = currentProblem[doc1]
            cv_fit = cv.fit([textDoc1])
            doc1Feats = cv_fit.transform([textDoc1]).toarray()[0]
            doc1Feats = doc1Feats/doc1Feats.sum()
            for doc2 in allDocs:
                if doc1 != doc2:
                    textDoc2 = currentProblem[doc2]
                    doc2Feats = cv_fit.transform([textDoc2]).toarray()[0]
                    if doc2Feats.sum() != 0:
                        doc2Feats = doc2Feats/doc2Feats.sum()
                    deltaVal = 0
                    for i in range(len(doc1Feats)):
                        deltaVal += abs(doc1Feats[i] - doc2Feats[i])
                    avg += deltaVal
                    docDict[doc2] = deltaVal
            avg = avg/(len(allDocs)-1)
            docDict["average"] = avg
            std = 0
            for doc2 in allDocs:
                if doc1 != doc2:
                    currentDelta = docDict[doc2]
                    std += (currentDelta - avg)**2
            std = np.sqrt(std/(len(allDocs)-1))
            docDict["std"] = std
            allInformation[doc1] = docDict
        allInfo[currentProbl] = allInformation
    return allInfo

def getSpatiumScores(allInfo):
    """ Given a dictionary with the combination of files and their distance
        score, return the actual scores and the set of problems, e.g their
        names. """
    allScores = {}
    problems = allInfo.keys()
    for prob in problems:
        currentInfo = allInfo[prob]
        allDocs = currentInfo.keys()
        scores = {}
        for doc1 in allDocs:
            doc1Dict = {}
            current = currentInfo[doc1]
            withinDocs = current.keys()
            avg = current["average"]
            std = current["std"]
            withinDocs.remove("std")
            withinDocs.remove("average")
            for doc2 in withinDocs:
                if std == 0.0:
                    doc1Dict[doc2] = avg - current[doc2]
                else:
                    doc1Dict[doc2] = (avg - current[doc2])/std
            scores[doc1] = doc1Dict
        allScores[prob] = scores
    return allScores, problems

def getSymmetricScores(problems, allInfo, allScores):
    """ Given a list with the names of a problem set, a dictionary with the
        combination of files and their distance score and a dictionary
        containing the scores for each combination of documents, return a tuple
        of dictionaries containing the symmetric scores, the maximum scores and
        minimum scores respectively. """
    allSymmetricScores = {}
    for prob in problems:
        currentInfo = allInfo[prob]
        allDocs = currentInfo.keys()
        scores = allScores[prob]
        symmetricScores = {}
        for doc1 in allDocs:
            doc1Dict = {}
            allScore = []
            current = scores[doc1]
            otherDocs = current.keys()
            for doc2 in otherDocs:
                score = abs(current[doc2]) + abs(scores[doc2][doc1])
                allScore.append(score)
                doc1Dict[doc2] = score
            symmetricScores[doc1] = doc1Dict
        allSymmetricScores[prob] = symmetricScores
    return allSymmetricScores

def getAllLinks(problems, allInfo, allSymmetricScores):
    """ Given a list with the names of a problem set, a dictionary with the
        combination of files and their distance score, all the symmetric scores,
        all the minimal scores, all the maximal scores for each document and
        the threshold for a link, return a dictionary with the possible links
        for each problem. """
    eachLinks = {}
    for prob in problems:
        currentInfo = allInfo[prob]
        allDocs = currentInfo.keys()
        symmetricScores = allSymmetricScores[prob]
        links = []
        currentKeys = []
        for doc1 in allDocs:
            for doc2 in allDocs:
                if doc1 != doc2:
                    # Symmetric, so can be found at one place
                    if (doc2, doc1) not in currentKeys:
                        score = symmetricScores[doc1][doc2]
                        currentKeys.append((doc1, doc2))
                        links.append({"document1": doc1, "document2": doc2,
                                    "score": score})
        eachLinks[prob] = sorted(links, key=lambda k: k["score"], reverse=True)
    return eachLinks

def to_graph(links):
    """ Given a list containing dictionaries for each link, create a network of
        these links. """
    G = nx.Graph()
    for link in links:
        doc1 = link["document1"]
        doc2 = link["document2"]
        G.add_nodes_from([doc1, doc2])
        G.add_edge(doc1, doc2)
    return G

def makeClusters(problems, eachLinks, allInfo):
    """ Given a list with the names of a problem set, a dictionary with lists as
        values per problem containing all the links sorted and a dictionary with
        the combination of files, return a list with clusters that are lists as
        well. """
    allClusters = {}
    for problemm in problems:
        currentProblem = eachLinks[problemm]
        current = allInfo[problemm]
        allDocs = current.keys()
        G = to_graph(currentProblem)
        clusters = list(nx.connected_components(G))
        newClusters = []
        processedDocuments = []
        for cluster in clusters:
            clust = []
            for doc in cluster:
                processedDocuments.append(doc)
                clust.append({"document": doc})
            newClusters.append(clust)
        remainingDocs = set(allDocs).difference(set(processedDocuments))
        for doc in remainingDocs:
            newClusters.append([{"document": doc}])
        allClusters[problemm] = newClusters
    return allClusters

def modularityGetClusters(problems, eachLinks, allInfo):
    allClusters = {}
    for problemm in problems:
        currentProblem = eachLinks[problemm]
        current = allInfo[problemm]
        allDocs = current.keys()
        G = nx.Graph()
        scores = list(set([link["score"] for link in currentProblem]))
        if len(scores) == 1 and scores[0] == 0:
            for doc in allDocs:
                G.add_node(doc)
        else:
            for doc in allDocs:
                G.add_node(doc)
            addWeights = [(link["document1"], link["document2"], link["score"]) for link in currentProblem]
            G.add_weighted_edges_from(addWeights)
        part = community.best_partition(G)
        values = [part.get(node) for node in G.nodes()]
        uniqueClusters = list(set(values))
        cluster = []
        for val in uniqueClusters:
            currentTerms = []
            for doc in allDocs:
                if part[doc] == val:
                    currentTerms.append({"document": doc})
            cluster.append(currentTerms)
        allClusters[problemm] = cluster
    return allClusters

def checkFolder(fileName):
    """ Given a fileName, check whether its path does exist. """
    if not os.path.exists(os.path.dirname(fileName)):
        try:
            os.makedirs(os.path.dirname(fileName))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def writeToFile(fileName, data):
    """ Given a filename and json data, store the json data at the given file
        name. """
    with open(fileName, "w") as f:
        json.dump(data, f)

def storeClustersLinks(problems, outputPath, allClusters, eachLinks):
    """ Given a list of all the names of the problems, the place where the
        findings will be stored, a list containing clusters and a list
        containing links, write the clusters and links to files belonging in the
        folder of the problem set. """
    for problem in problems:
        fileName1 = outputPath + problem + "/clustering.json"
        fileName2 = outputPath + problem + "/ranking.json"
        checkFolder(fileName1)
        writeToFile(fileName1, allClusters[problem])
        checkFolder(fileName2)
        writeToFile(fileName2, eachLinks[problem])

def applySpatium(firstPath, outputPath, option, k, n, modeFeats):
    """ Given a path with the data, a path to where the findings will be stored,
        the threshold for links, an option how to handle unique terms and the value
        of n in the character n-grams, store the clusters and links for each
        problem. """
    newDataDict = extractData(firstPath)
    allInfo = getSpatiumDistances(newDataDict, option, k, n, modeFeats)
    allScores, problems = getSpatiumScores(allInfo)
    allSymmetricScores = getSymmetricScores(problems, allInfo, allScores)
    eachLinks = getAllLinks(problems, allInfo, allSymmetricScores)
    #allClusters = makeClusters(problems, eachLinks, allInfo)
    allClusters = modularityGetClusters(problems, eachLinks, allInfo)
    probls = eachLinks.keys()
    newLinks = {}
    for p in probls:
        linkss = []
        current = eachLinks[p]
        maxVal = current[0]["score"]
        for l in current:
            if maxVal == 0.0:
                newScore = l["score"]
            else:
                newScore = l["score"]/maxVal
            linkss.append({"document1": l["document1"], "document2": l["document2"], "score": newScore})
        newLinks[p] = linkss
    storeClustersLinks(problems, outputPath, allClusters, newLinks)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("spatium2.py -i EVAL_DIRECTORY -o OUTPUT_DIRECTORY")
    firstPath = sys.argv[2]
    outputPath = sys.argv[4]
    option = "top"
    applySpatium(firstPath, outputPath, option, 350, 4, "char")
    # nVals = [1, 2, 3, 4, 5, 6]
    # kVals = [1, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    # for n in nVals:
    #     for k in kVals:
    #         newPath = outputPath + "n" + str(n) + "c/k" + str(k) + "/"
    #         applySpatium(firstPath, newPath, option, k, n, "char")
    #         newPath1 = outputPath + "n" + str(n) + "w/k" + str(k) + "/"
    #         applySpatium(firstPath, newPath1, option, k, n, "word")
    #         print "finished", n, k
