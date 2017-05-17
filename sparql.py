import urllib, json

def sparqlQuery(query, baseURL, format="application/json"):
	params={
		"default-graph": "",
		"should-sponge": "soft",
		"query": query,
		"debug": "on",
		"timeout": "",
		"format": format,
		"save": "display",
		"fname": ""
	}
	querypart=urllib.urlencode(params)
	response = urllib.urlopen(baseURL,querypart).read()
	try:
		jsonVal = json.loads(response)
		return jsonVal
	except ValueError as e:
		return {}



def composeSparqlQuery(annotations, property):
	entities = []
	cProp = []
	for ann in annotations:
		if ann[1] == "EB":
			if len(cProp) != 0:
				entities.append(cProp)
				cProp = []
			cProp.append(ann[0])
		if ann[1] == "EI":
			cProp.append(ann[0])
	if len(cProp) != 0:
		entities.append(cProp)

	if not property:
		name = " ".join(entities[0]).title()
		query = "PREFIX dbo: <http://dbpedia.org/ontology/> " \
				"PREFIX dbp: <http://dbpedia.org/property/> " \
				"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
				'SELECT distinct ?property ' \
				'WHERE { ?entity rdfs:label ?name . ' \
				"?entity a <http://dbpedia.org/ontology/Person> . " \
				"?entity dbo:abstract ?property . " \
				"FILTER langMatches(lang(?name ), \"EN\").  " \
				"FILTER langMatches(lang(?property ), \"EN\").  " \
				'FILTER contains(?name, "%s") . ' % name
		query = query + " ?entity %s ?property . " % property
		query = query + "}"
	else:
		name = " ".join(entities[0]).title()
		query = "PREFIX dbo: <http://dbpedia.org/ontology/> " \
				"PREFIX dbp: <http://dbpedia.org/property/> " \
				"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
				'SELECT distinct ?property ?property2 ' \
				'WHERE { ?entity rdfs:label ?name . ' \
				'?entity a <http://dbpedia.org/ontology/Person> . ' \
				'FILTER contains(?name, "%s") . ' % name
		query = query + " OPTIONAL { ?entity dbp:%s ?property } . " % property
		query = query + " OPTIONAL { ?entity dbo:%s ?property2 } . " % property
		query = query + "}"
	return query

def getPropertyResult(queryResponse):
	props = []
	if "results" in queryResponse:
		resultsJ = queryResponse["results"]
		if "bindings" in resultsJ:
			bindingsJ = resultsJ["bindings"]
			for propJ in bindingsJ:
				if "property" in propJ and "value" in propJ["property"]:
					props.append(propJ["property"]["value"])
				if "property2" in propJ and "value" in propJ["property2"]:
					props.append(propJ["property2"]["value"])
	return props

def launchSparqlQuery(query, url="http://141.85.227.62:8890/sparql"):
	data = sparqlQuery(query, url)
	return data
