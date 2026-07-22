CREATE TABLE handle (
    ROWID INTEGER PRIMARY KEY,
    id TEXT
);

CREATE TABLE chat (
    ROWID INTEGER PRIMARY KEY,
    chat_identifier TEXT
);

CREATE TABLE message (
    ROWID INTEGER PRIMARY KEY,
    guid TEXT,
    text TEXT,
    attributedBody BLOB,
    is_from_me INTEGER DEFAULT 0,
    handle_id INTEGER,
    date INTEGER,
    service TEXT,
    item_type INTEGER DEFAULT 0,
    associated_message_type INTEGER DEFAULT 0,
    balloon_bundle_id TEXT,
    is_deleted INTEGER DEFAULT 0,
    date_retracted INTEGER DEFAULT 0
);

CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
CREATE TABLE attachment (ROWID INTEGER PRIMARY KEY, filename TEXT);
CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);

INSERT INTO handle VALUES (1, 'alice@example.com');
INSERT INTO handle VALUES (2, '+12125550199');
INSERT INTO chat VALUES (1, 'chat-alice');
INSERT INTO chat VALUES (2, 'group-secret');
INSERT INTO chat_handle_join VALUES (1, 1);
INSERT INTO chat_handle_join VALUES (2, 1);
INSERT INTO chat_handle_join VALUES (2, 2);

INSERT INTO message VALUES (
    1, 'guid-1', 'email alice@example.com or call +1 (212) 555-0199 https://example.com',
    NULL, 0, 1, 700000000, 'iMessage', 0, 0, NULL, 0, 0
);
INSERT INTO message VALUES (
    2, 'guid-2', 'sounds good 😊', NULL, 1, NULL, 700000060, 'iMessage', 0, 0, NULL, 0, 0
);
INSERT INTO message VALUES (
    3, 'guid-3', NULL, NULL, 0, 1, 700000180, 'iMessage', 0, 0, NULL, 0, 0
);
INSERT INTO message VALUES (
    4, 'guid-4', 'Loved “sounds good”', NULL, 0, 1, 700000200, 'iMessage', 0, 2000,
    NULL, 0, 0
);
INSERT INTO message VALUES (
    5, 'guid-5', 'Alice changed the group name', NULL, 0, 1, 700000300, 'iMessage', 1, 0,
    NULL, 0, 0
);
INSERT INTO message VALUES (
    6, 'guid-6', NULL, NULL, 0, 1, 700000400, 'iMessage', 0, 0, NULL, 0, 0
);
INSERT INTO message VALUES (
    7, 'guid-7', 'removed secret', NULL, 1, NULL, 700000500, 'iMessage', 0, 0, NULL, 1, 0
);
INSERT INTO message VALUES (
    8, 'guid-8', 'look at this', NULL, 0, 2, 700000600, 'iMessage', 0, 0, NULL, 0, 0
);

INSERT INTO chat_message_join VALUES (1, 1);
INSERT INTO chat_message_join VALUES (1, 2);
INSERT INTO chat_message_join VALUES (1, 3);
INSERT INTO chat_message_join VALUES (1, 4);
INSERT INTO chat_message_join VALUES (1, 5);
INSERT INTO chat_message_join VALUES (1, 6);
INSERT INTO chat_message_join VALUES (1, 7);
INSERT INTO chat_message_join VALUES (2, 8);

INSERT INTO attachment VALUES (1, '/private/photo.jpg');
INSERT INTO attachment VALUES (2, '/private/document.pdf');
INSERT INTO message_attachment_join VALUES (6, 1);
INSERT INTO message_attachment_join VALUES (8, 2);
