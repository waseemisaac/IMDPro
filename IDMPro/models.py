from django.db import models

class ModelResult(models.Model):
    model_name = models.CharField(max_length=100)
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()

    def __str__(self):
        return f"{self.model_name} Results"
