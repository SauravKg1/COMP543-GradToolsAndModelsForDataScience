-- I HAVE NAMED MY DATABASE AS DB_SG275

-- Who has seen a flower at Alaska Flat?

SELECT DISTINCT PERSON FROM SIGHTINGS WHERE LOCATION = 'Alaska Flat';


-- Who has seen the same flower at both Moreland Mill and at Steve Spring?

SELECT DISTINCT s1.PERSON
FROM SIGHTINGS s1
JOIN SIGHTINGS s2
    ON s1.PERSON = s2.PERSON
   AND s1.NAME = s2.NAME   -- same flower
WHERE s1.LOCATION = 'Moreland Mill'
  AND s2.LOCATION = 'Steve Spring';


-- What is the scientific name for each of the different flowers that have
--     been sighted by either Michael or Robert below 7250 feet in elevation?

SELECT DISTINCT f.GENUS, f.SPECIES
FROM SIGHTINGS s
JOIN FLOWERS f
    ON s.NAME = f.COMNAME
JOIN PEOPLE p
    ON s.PERSON = p.PERSON
JOIN FEATURES fe
    ON s.LOCATION = fe.LOCATION
WHERE p.PERSON IN ('Michael', 'Robert')
  AND fe.ELEV < 7250;


-- Which maps hold a location where someone has seen Alpine penstemon in June?

SELECT DISTINCT fe.MAP
FROM SIGHTINGS s
JOIN FEATURES fe ON s.LOCATION = fe.LOCATION
WHERE s.NAME = 'Alpine penstemon'
  AND MONTH(s.SIGHTED) = 6;


-- Which genus have more than one species recorded in the SSWC database?

SELECT f.GENUS
FROM FLOWERS f
GROUP BY f.GENUS
HAVING COUNT(DISTINCT f.SPECIES) > 1;


-- How many mines are on the Claraville map?

SELECT COUNT(*) AS NumMines
FROM FEATURES
WHERE MAP = 'Claraville'
  AND CLASS = 'Mine';


-- What is the furthest north location that James has seen a flower?
--     “Furthest north” means highest latitude.

SELECT TOP 1 WITH TIES fe.LOCATION
FROM SIGHTINGS s
JOIN FEATURES fe ON s.LOCATION = fe.LOCATION
WHERE s.PERSON = 'James'
ORDER BY fe.LATITUDE DESC;


-- Who has not seen a flower at a location of class Spring?

SELECT p.PERSON
FROM PEOPLE p
WHERE NOT EXISTS (
    SELECT 1
    FROM SIGHTINGS s
    JOIN FEATURES fe ON s.LOCATION = fe.LOCATION
    WHERE s.PERSON = p.PERSON
      AND fe.CLASS = 'Spring'
);


-- Who has seen flowers at the least distinct locations, and how many distinct
--     flowers was that?

WITH per_person AS (
    SELECT p.PERSON,
           COUNT(DISTINCT s.LOCATION) AS DistinctLocations,
           COUNT(DISTINCT s.NAME)     AS DistinctFlowers
    FROM PEOPLE p
    LEFT JOIN SIGHTINGS s ON p.PERSON = s.PERSON
    GROUP BY p.PERSON
)
SELECT PERSON, DistinctLocations, DistinctFlowers
FROM per_person
WHERE DistinctLocations = (SELECT MIN(DistinctLocations) FROM per_person);


-- For those people who have seen all of the flowers in the SSWC database,
--     what was the date at which they saw their last unseen flower?
--     In other words, at which date did they finish observing all of the
--     flowers in the database?

DECLARE @TotalFlowers INT = (SELECT COUNT(*) FROM FLOWERS);

WITH first_seen AS (
    SELECT s.PERSON, s.NAME, MIN(s.SIGHTED) AS FirstSeenDate
    FROM SIGHTINGS s
    GROUP BY s.PERSON, s.NAME
),
person_counts AS (
    SELECT PERSON, COUNT(*) AS FlowersSeenCount
    FROM first_seen
    GROUP BY PERSON
)
SELECT fs.PERSON,
       MAX(fs.FirstSeenDate) AS FinishedAllOn
FROM first_seen fs
JOIN person_counts pc ON fs.PERSON = pc.PERSON
WHERE pc.FlowersSeenCount = @TotalFlowers
GROUP BY fs.PERSON;


-- For Tim, compute the fraction of his sightings on a per-month basis.
--     For example, we might get {(September, .12), (October, .74),
--         (November, .14)}. The fractions should add up to one across
-- all months.

WITH TimSightings AS (
    SELECT MONTH(SIGHTED) AS MonthNum,
           DATENAME(month, SIGHTED) AS MonthName
    FROM SIGHTINGS
    WHERE PERSON = 'Tim'
),
Total AS (
    SELECT COUNT(*) AS TotalCount FROM TimSightings
)
SELECT t.MonthName,
       COUNT(*) * 1.0 / tot.TotalCount AS Fraction
FROM TimSightings t
CROSS JOIN Total tot
GROUP BY t.MonthName, t.MonthNum, tot.TotalCount
ORDER BY t.MonthNum;


-- Whose set of flower sightings is most similar to Michael’s? Set
--     similarity is here defined in terms of the Jaccard Index, where JI (A, B)
--     for two sets A and B is (size of the intersection of A and B) / (size
--     of the union of A and B). A larger Jaccard Index means more similar.

WITH MichaelFlowers AS (
    SELECT DISTINCT NAME
    FROM SIGHTINGS
    WHERE PERSON = 'Michael'
),
PersonFlowers AS (
    SELECT DISTINCT PERSON, NAME
    FROM SIGHTINGS
),
PersonCounts AS (
    SELECT PERSON, COUNT(*) AS PersonCount
    FROM PersonFlowers
    GROUP BY PERSON
),
MichaelCount AS (
    SELECT COUNT(*) AS MCount FROM MichaelFlowers
),
Intersections AS (
    SELECT pf.PERSON, COUNT(*) AS InterCount
    FROM PersonFlowers pf
    JOIN MichaelFlowers mf ON pf.NAME = mf.NAME
    GROUP BY pf.PERSON
)
SELECT TOP 1 WITH TIES
       pc.PERSON,
       CAST(ISNULL(ic.InterCount, 0) AS FLOAT) /
       (pc.PersonCount + mc.MCount - ISNULL(ic.InterCount, 0)) AS JaccardIndex
FROM PersonCounts pc
LEFT JOIN Intersections ic ON pc.PERSON = ic.PERSON
CROSS JOIN MichaelCount mc
WHERE pc.PERSON <> 'Michael'
ORDER BY JaccardIndex DESC;