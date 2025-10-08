# users/models.py
from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=100)
    encoding = models.BinaryField()  # Pickled face encoding
    created_at = models.DateTimeField(auto_now_add=True)

    # models.py
# from django.db import models

class EmotionRecord(models.Model):
    name = models.CharField(max_length=100)
    emotion = models.CharField(max_length=50)
    count = models.IntegerField(default=1)
    first_detected = models.DateTimeField(auto_now_add=True)
    last_detected = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} - {self.emotion} ({self.count})"

