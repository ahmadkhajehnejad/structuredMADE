import keras.backend as K
import tensorflow as tf
import config
import numpy as np


def _logistic_cdf(scaled_centered_d):
    return tf.nn.sigmoid(scaled_centered_d)

def logistic_loss(y_true, y_pred):
    tmp_sz = config.num_mixture_components * config.graph_size
    mu_pred, logScale_pred, logpi_unnormalized_pred = y_pred[:, :tmp_sz], y_pred[:, tmp_sz:2 * tmp_sz], y_pred[:,
                                                                                                        2 * tmp_sz:]

    mu_pred = K.reshape(mu_pred, [-1, config.num_mixture_components, config.graph_size])
    logScale_pred = K.reshape(logScale_pred, [-1, config.num_mixture_components, config.graph_size])
    logpi_unnormalized_pred = K.reshape(logpi_unnormalized_pred,
                                        [-1, config.num_mixture_components, config.graph_size])
    logpi_pred = logpi_unnormalized_pred - K.tile(K.logsumexp(logpi_unnormalized_pred, axis=1, keepdims=True),
                                                  [1, config.num_mixture_components, 1])

    logScale_pred = tf.maximum(logScale_pred, config.min_logScale)

    inv_s_pred = K.exp(-logScale_pred)

    y_true_tiled = K.tile(K.expand_dims(y_true, 1), [1, config.num_mixture_components, 1])

    if config.scale_negOne_to_posOne:
        half_delta = (1. / 255.)
    else:
        half_delta = (0.5 / 255.)

    plus_in = inv_s_pred * (y_true_tiled + half_delta - mu_pred)
    min_in = inv_s_pred * (y_true_tiled - half_delta - mu_pred)

    delta_cdf = _logistic_cdf(plus_in) - _logistic_cdf(min_in)

    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)  # log probability for edge case of 255 (before scaling)

    mid_in = inv_s_pred * (y_true_tiled - mu_pred)
    log_pdf_mid = mid_in - logScale_pred - 2. * tf.nn.softplus(
        mid_in)  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    if config.scale_negOne_to_posOne:
        left_boundary = -0.999
        right_boundary = 0.999
        d_ = 127.5
    else:
        left_boundary = 0.001
        right_boundary = 0.999
        d_ = 255.

    if config.robust:
        tmp = tf.where(y_true_tiled < left_boundary, log_cdf_plus,
                       tf.where(y_true_tiled > right_boundary, log_one_minus_cdf_min,
                                tf.where(delta_cdf > 1e-5, tf.log(tf.maximum(delta_cdf, 1e-12)),
                                         log_pdf_mid - np.log(d_))))
    else:
        tmp = tf.where(y_true_tiled < left_boundary, log_cdf_plus,
                       tf.where(y_true_tiled > right_boundary, log_one_minus_cdf_min,
                                tf.log(tf.maximum(delta_cdf, 1e-12))))

    tmp = tmp + logpi_pred

    tmp = K.logsumexp(tmp, axis=1)
    return -1 * K.sum(tmp, axis=1)
