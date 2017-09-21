from __future__ import division
import os
import networkx as nx
from networkx.algorithms.components.connected import connected_components
import glob
import community
import json
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math
import sys
from collections import Counter
import random
import glob
from pprint import pprint
import pickle

def flatten(l):
	return [item for sublist in l for item in sublist]

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
				newData[probl] = problem.strip().lower()
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

def euclideanSquaredNormalized(v1, v2):
	""" Given two vectors of the same length, calculate their normalized squared euclidean distance. """
	sumVal = 0
	size = v1.size
	for i in range(size):
		sumVal += ((v1[i] - v2[i])**2)/size
	return sumVal

def cosineSimilarity(v1, v2):
	""" Given two vectors of the same length, calculate their cosine similarity. """
	norm1 = np.linalg.norm(v1)
	norm2 = np.linalg.norm(v2)
	if norm1 == 0.0 or norm2 == 0.0:
		return 0
	else:
		return np.dot(v1, v2)/(norm1 * norm2)

def minMaxDistance(v1, v2):
	""" Given two vectors of the same length, calculate their min-max distance. """
	minVal = 0
	maxVal = 0
	size = v1.size
	for i in range(size):
		minVal += min(v1[i], v2[i])
		maxVal += max(v1[i], v2[i])
	return minVal/maxVal

def imposterMethod(newDataDict, option, k, n, modeFeats, amountRuns, imposters):
	""" Given a dictionary containing the text for each document within each
		problem set, an option for unique terms/top features, an amount for the
		top k features, the value n for the character n-grams and whether it is
		character or words n-grams with the given n, return the distances
		for each pair of documents. """
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
					impostVal = imposterFunc(doc1Feats, doc2Feats, imposters,
											 amountRuns, k, cv_fit, doc1, doc2, currentProbl)
					docDict[doc2] = impostVal
			allInformation[doc1] = docDict
		allInfo[currentProbl] = allInformation
		print("One problem done")
	return allInfo

def imposterFunc(doc1Feats, doc2Feats, imposters, amountRuns, amountFeatures, cv_fit,
				 doc1, doc2, problem):
	score = 0
	listF = range(0, len(doc1Feats) - amountFeatures)
	if listF == []:
		listF = [0]
	featureSets = [random.choice(listF) for _ in range(amountRuns)]
	impostersDoc1 = imposters[problem][doc1]
	impostersDoc1 = [cv_fit.transform([text]).toarray()[0] for text in impostersDoc1]
	impostersDoc2 = imposters[problem][doc2]
	impostersDoc2 = [cv_fit.transform([text]).toarray()[0] for text in impostersDoc2]
	for setVal in featureSets:
		useDoc1 = doc1Feats[setVal:setVal+amountFeatures]
		useDoc2 = doc2Feats[setVal:setVal+amountFeatures]
		sim = cosineSimilarity(useDoc1, useDoc2)
		for impost in impostersDoc1:
			sim1 = cosineSimilarity(useDoc1, impost[setVal:setVal+amountFeatures])
			if sim > sim1:
				score += 1
		for impost in impostersDoc2:
			sim1 = cosineSimilarity(useDoc2, impost[setVal:setVal+amountFeatures])
			if sim > sim1:
				score += 1
	return score

def getAllLinks(problems, allInfo, threshold):
	""" Given a list of all problem sets, a dictionary with the combination of
		files and their distancescore and a threshold, return the links above
		the threshold. """
	eachLinks = {}
	for prob in problems:
		currentInfo = allInfo[prob]
		allDocs = currentInfo.keys()
		links = []
		currentKeys = []
		for doc1 in allDocs:
			for doc2 in allDocs:
				if doc1 != doc2:
					# Symmetric, so can be found at one place
					if (doc2, doc1) not in currentKeys:
						score = (currentInfo[doc1][doc2] + currentInfo[doc2][doc1])/2
						currentKeys.append((doc1, doc2))
						links.append({"document1": doc1, "document2": doc2,
									 "score": score})
		eachLinks[prob] = sorted(links, key=lambda k: k["score"], reverse=True)
	return eachLinks

def findAllImposters(amountImposters, option, k, n, mode, newDataDict):
	imposts = {}
	enImpost, duImpost, grImpost = getAllImposters()
	cv = CountVectorizer(encoding="unicode", analyzer=lambda text:
						 createFeaturesNGrams(text, n, k, option, mode))
	problems = newDataDict.keys()
	for p in problems:
		current = newDataDict[p]
		docs = current.keys()
		for doc in docs:
			currentText = current[doc]
			allImposts = []
			currentDict = {}
			for impost in enImpost:
				cv_fit = cv.fit([currentText + " " + impost])
				docFeats = cv_fit.transform([currentText]).toarray()[0]
				docFeats = docFeats/docFeats.sum()
				impostFeats = cv_fit.transform([impost]).toarray()[0]
				impostFeats = impostFeats/impostFeats.sum()
				sim = cosineSimilarity(docFeats, impostFeats)
				allImposts.append((impost, sim))
			for impost in duImpost:
				cv_fit = cv.fit([currentText + " " + impost])
				docFeats = cv_fit.transform([currentText]).toarray()[0]
				docFeats = docFeats/docFeats.sum()
				impostFeats = cv_fit.transform([impost]).toarray()[0]
				impostFeats = impostFeats/impostFeats.sum()
				sim = cosineSimilarity(docFeats, impostFeats)
				allImposts.append((impost, sim))
			for impost in grImpost:
				cv_fit = cv.fit([currentText + " " + impost])
				docFeats = cv_fit.transform([currentText]).toarray()[0]
				docFeats = docFeats/docFeats.sum()
				impostFeats = cv_fit.transform([impost]).toarray()[0]
				impostFeats = impostFeats/impostFeats.sum()
				sim = cosineSimilarity(docFeats, impostFeats)
				allImposts.append((impost, sim))
			sortedImpost = sorted(allImposts, key=lambda tup: tup[1], reverse=True)
			currentDict[doc] = [impost[0] for impost in sortedImpost[:amountImposters]]
		imposts[p] = currentDict
	return imposts


def getAllImposters():
	infoFile = glob.glob("imposters/info.json")[0]
	with open(infoFile) as data_file:
		data = json.load(data_file)
	imposterEn = [entry["folder"] for entry in data if entry["language"] == "en"]
	imposterDu = [entry["folder"] for entry in data if entry["language"] == "nl"]
	imposterGr = [entry["folder"] for entry in data if entry["language"] == "gr"]
	imposterDocsEn = flatten([glob.glob("imposters/" + imposterEn[i] + "/*") for i in range(len(imposterEn))])
	imposterDocsDu = flatten([glob.glob("imposters/" + imposterDu[i] + "/*") for i in range(len(imposterDu))])
	imposterDocsGr = flatten([glob.glob("imposters/" + imposterGr[i] + "/*") for i in range(len(imposterGr))])
	imposterTextEn = []
	for subProblem in imposterDocsEn:
		with open(subProblem) as curData:
			problem = unicode(curData.read(), "utf-8")
			text = problem.strip().lower()
			imposterTextEn.append(text)
	imposterTextDu = []
	for subProblem in imposterDocsDu:
		with open(subProblem) as curData:
			problem = unicode(curData.read(), "utf-8")
			text = problem.strip().lower()
			imposterTextDu.append(text)
	imposterTextGr = []
	for subProblem in imposterDocsGr:
		with open(subProblem) as curData:
			problem = unicode(curData.read(), "utf-8")
			text = problem.strip().lower()
			imposterTextGr.append(text)
	return imposterTextEn, imposterTextDu, imposterTextGr

def applyImposter(firstPath, outputPath, threshold, amountRuns, option, k, n, mode, amountImposters):
	""" Given a path with the data, a path to where the findings will be stored,
		the threshold for links, an option how to handle unique terms and the value
		of n in the character n-grams, store the clusters and links for each
		problem. """
	newDataDict = extractData(firstPath)
	#allImposters = findAllImposters(amountImposters, option, k, n, mode, newDataDict)
	with open('outcomeImposters.txt', 'rb') as handle:
  		allImposters = pickle.loads(handle.read())
	allInfo = imposterMethod(newDataDict, option, k, n, mode, amountRuns, allImposters)
	problems = allInfo.keys()
	eachLinks = getAllLinks(problems, allInfo, threshold)
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
		print("imposter.py -i EVAL_DIRECTORY -o OUTPUT_DIRECTORY")
	firstPath = sys.argv[2]
	outputPath = sys.argv[4]
	option = "normal"
	applyImposter(firstPath, outputPath, 0.2, 100, option, 300, 4, "char", 25)
