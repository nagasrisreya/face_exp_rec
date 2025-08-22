# users/models.py
from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=100)
    encoding = models.BinaryField()  # Pickled face encoding
    created_at = models.DateTimeField(auto_now_add=True)
