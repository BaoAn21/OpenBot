package org.openbot.objectbox.entity;


import io.objectbox.annotation.Entity;
import io.objectbox.annotation.Id;

@Entity
public class LogEntry {
    @Id
    public long id;
    public String text;
    public long timestamp;


    public LogEntry(String text, long timestamp) {
        this.text = text;
        this.timestamp = timestamp;
    }
    public LogEntry() {
        // ObjectBox requires a no-arg constructor to construct entities.
    }

    @Override
    public String toString() {
        return "LogEntry{" + "id=" + id + ", text='" + text + '\'' + ", timestamp=" + timestamp + '}';
    }
}