DROP PROCEDURE IF EXISTS ConnectedComponents;
GO

CREATE PROCEDURE ConnectedComponents
AS
BEGIN
    IF OBJECT_ID('temp_edges') IS NOT NULL DROP TABLE temp_edges;
    SELECT paperID, citedPaperID INTO temp_edges FROM edges
    UNION
    SELECT citedPaperID, paperID FROM edges;

    IF OBJECT_ID('components') IS NOT NULL DROP TABLE components;
    CREATE TABLE components (
        paperID INT,
        compID INT
    );

    DECLARE @compID INT = 0;

    DECLARE curs CURSOR FOR SELECT paperID FROM nodes;
    DECLARE @start INT;

    OPEN curs;
    FETCH NEXT FROM curs INTO @start;
    WHILE @@FETCH_STATUS = 0
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM components WHERE paperID = @start)
        BEGIN
            SET @compID = @compID + 1;

            IF OBJECT_ID('queue') IS NOT NULL DROP TABLE queue;
            CREATE TABLE queue (paperID INT PRIMARY KEY);

            INSERT INTO queue VALUES (@start);

            WHILE EXISTS (SELECT 1 FROM queue)
            BEGIN
                INSERT INTO components (paperID, compID)
                SELECT q.paperID, @compID
                FROM queue q
                WHERE NOT EXISTS (SELECT 1 FROM components c WHERE c.paperID = q.paperID);

                DELETE q
                FROM queue q
                JOIN components c ON q.paperID = c.paperID;

                INSERT INTO queue
                SELECT DISTINCT e.citedPaperID
                FROM temp_edges e
                JOIN components c ON e.paperID = c.paperID AND c.compID = @compID
                WHERE NOT EXISTS (SELECT 1 FROM components cc WHERE cc.paperID = e.citedPaperID);
            END
        END

        FETCH NEXT FROM curs INTO @start;
    END
    CLOSE curs;
    DEALLOCATE curs;

    SELECT c.compID, n.paperID, n.paperTitle
    FROM components c
    JOIN nodes n ON c.paperID = n.paperID
    WHERE c.compID IN (
        SELECT compID FROM components GROUP BY compID HAVING COUNT(*) BETWEEN 5 AND 10
    )
    ORDER BY c.compID, n.paperID;
END;
GO

EXEC ConnectedComponents;
