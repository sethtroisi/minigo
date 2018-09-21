import tensorflow as tf

import dual_net
import features as features_lib
import go


dual_net.flags.FLAGS([""])
sess = tf.Session()
save_file = "saved_models/000721-eagle"

features, labels = dual_net.get_inference_input()
p_out, v_out, logits, shared = dual_net.model_inference_fn(features, False)
tf.train.Saver().restore(sess, save_file)

predictions = {
    'policy_output': p_out,
    'value_output': v_out,
    'shared': shared
}

p = go.Position()
processed = [features_lib.extract_features(p)]
res = sess.run(predictions,
               feed_dict={features: processed})

print(res['shared'][0])
