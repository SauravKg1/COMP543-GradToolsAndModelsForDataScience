
CREATE TABLE FLOWERS (
	flow_id INTEGER, 
	genus VARCHAR(30), 
	species VARCHAR(30), 
	comname VARCHAR(30), 
	PRIMARY KEY (comname));

CREATE TABLE SIGHTINGS (
	sight_id INTEGER, 
	name VARCHAR(30), 
	person VARCHAR(30), 
	location VARCHAR(30), 
	sighted DATETIME, 
	PRIMARY KEY (name, person, location, sighted));

CREATE TABLE PEOPLE (
	person_id INTEGER, 
	person VARCHAR(30), 
	PRIMARY KEY (person_id));

CREATE TABLE FEATURES (
	loc_id INTEGER, 
	location VARCHAR(30), 
	class VARCHAR(30), 
	latitude INTEGER, 
	longitude INTEGER, 
	map VARCHAR(30), 
	elev INTEGER,  
	PRIMARY KEY (location));
