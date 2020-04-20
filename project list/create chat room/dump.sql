BEGIN TRANSACTION;
CREATE TABLE book(
        title,
        author,
        published
    );
INSERT INTO "book" VALUES('Dirk Gently''s Holistic Detective Agency','Douglas Adams',1987);
CREATE TABLE person(
        firstname,
        lastname,
        age
    );
COMMIT;
