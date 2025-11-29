DROP PROCEDURE IF EXISTS ComputePR;
GO

CREATE PROCEDURE ComputePR
AS
BEGIN
    DECLARE @d FLOAT = 0.85;
    DECLARE @delta FLOAT = 1.0;
    DECLARE @n INT;

    SELECT @n = COUNT(*) FROM nodes;

    IF OBJECT_ID('pagerank') IS NOT NULL DROP TABLE pagerank;
    CREATE TABLE pagerank (paperID INT PRIMARY KEY, rank FLOAT);

    INSERT INTO pagerank
    SELECT paperID, 1.0/@n FROM nodes;

    IF OBJECT_ID('outdeg') IS NOT NULL DROP TABLE outdeg;
    SELECT paperID, COUNT(*) AS outdeg INTO outdeg FROM edges GROUP BY paperID;

    WHILE @delta > 0.01
    BEGIN
        IF OBJECT_ID('newrank') IS NOT NULL DROP TABLE newrank;
        CREATE TABLE newrank (paperID INT PRIMARY KEY, rank FLOAT);

        INSERT INTO newrank
        SELECT n.paperID,
               (1.0 - @d)/@n +
               @d * (
                   ISNULL((
                       SELECT SUM(pr.rank / od.outdeg)
                       FROM edges e
                       JOIN pagerank pr ON e.paperID = pr.paperID
                       JOIN outdeg od ON e.paperID = od.paperID
                       WHERE e.citedPaperID = n.paperID
                   ), 0)
               )
        FROM nodes n;

        DECLARE @sinkRank FLOAT = (
            SELECT SUM(pr.rank)
            FROM pagerank pr
            WHERE pr.paperID NOT IN (SELECT paperID FROM outdeg)
        );
        UPDATE newrank
        SET rank = rank + @d * @sinkRank / @n;

        SELECT @delta = SUM(ABS(n.rank - p.rank))
        FROM newrank n
        JOIN pagerank p ON n.paperID = p.paperID;

        DELETE FROM pagerank;
        INSERT INTO pagerank SELECT * FROM newrank;
    END;

    PRINT 'Top 10 Papers by PageRank:';
    SELECT TOP 10 n.paperID, n.paperTitle, p.rank
    FROM pagerank p
    JOIN nodes n ON p.paperID = n.paperID
    ORDER BY p.rank DESC;
END;
GO
EXEC ComputePR;
