def iou(predicted_mask_batch, target_mask_batch):
    """ Считает среднее IoU для всех элементов батча """
    # Площадь пересечения в пикселях

    intersection = (predicted_mask_batch & target_mask_batch).sum(dim=(1, 2, 3))
    # Площадь объединения в пикселях
    union = (predicted_mask_batch | target_mask_batch).sum(dim=(1, 2, 3))
    iou = (intersection / union).mean()
    return iou
