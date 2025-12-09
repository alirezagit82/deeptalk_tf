from ..helper import *
from .compgcn_conv import CompGCNConv
from .compgcn_conv_basis import CompGCNConvBasis



class BaseModel(tf.keras.Model):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p = params
		self.bce_loss_fn = tf.keras.losses.BinaryCrossentropy()

	def loss(self, pred, true_label):
		return self.bce_loss_fn(true_label, pred)




class CompGCNBase(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(CompGCNBase, self).__init__(params)

        self.edge_index = edge_index
        self.edge_type = edge_type
        
        # Determine GCN dimension
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim

        # Initialize entity embeddings as a trainable variable
        self.init_embed = get_param((self.p.num_ent, self.p.init_dim), name='entity_embed')

        # Initialize relation embeddings
        if self.p.num_bases > 0:
            self.init_rel = get_param((self.p.num_bases, self.p.init_dim), name='rel_basis_embed')
        else:
            if self.p.score_func == 'transe':
                self.init_rel = get_param((num_rel, self.p.init_dim), name='rel_embed')
            else:
                self.init_rel = get_param((num_rel * 2, self.p.init_dim), name='rel_embed')

        # Initialize convolutional layers
        if self.p.num_bases > 0:
            self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
            self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, self.p.opn) if self.p.gcn_layer == 2 else None
        else:
            self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim, self.p.opn)
            self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, self.p.opn) if self.p.gcn_layer == 2 else None

        # Initialize bias as a trainable variable
        self.bias = self.add_weight(
            name='bias',
            shape=(self.p.num_ent,),
            initializer='zeros',
            trainable=True
        )

    def call(self, sub, rel, drop1, drop2):
        """
        The forward pass of the model.
        Note: The original method was `forward_base`, which is renamed to `call` in Keras.
        """
        # Prepare relation embeddings based on the score function
        if self.p.score_func != 'transe':
            r = self.init_rel
        else:
            # For TransE, we concatenate relation and its inverse
            r = tf.concat([self.init_rel, -self.init_rel], axis=0)

        # First convolution layer
        x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
        x = drop1(x)

        # Second convolution layer (if it exists)
        if self.p.gcn_layer == 2:
            x, r = self.conv2(x, self.edge_index, self.edge_type, rel_embed=r)
            x = drop2(x)

        # Gather embeddings for the specific subjects and relations in the batch
        sub_emb = tf.gather(x, indices=sub, axis=0)
        rel_emb = tf.gather(r, indices=rel, axis=0)

        return sub_emb, rel_emb, x

class CompGCN_TransE(CompGCNBase):
    def __init__(self, edge_index, edge_type, params=None):
        super().__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = tf.keras.layers.Dropout(self.p.hid_drop)

    def call(self, sub, rel):

        sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb	= sub_emb + rel_emb
        obj_emb_expanded = tf.expand_dims(obj_emb, axis=1)

        distances = tf.norm(obj_emb_expanded - all_ent, ord=1, axis=2)
        x = self.p.gamma - distnace

        score = tf.sigmoid(x)
        return socre
    
		# x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)		
		# score	= torch.sigmoid(x)

		# return score

class CompGCN_DistMult(CompGCNBase):
    def __init__(self, edge_index, edge_type, params=None):
        super().__init__(params.num_rel, params)
        self.drop = tf.keras.layers.Dropout(self.p.hid_drop)
        self.bias = self.add_weight(
            name='bias', 
            shape=(self.p.num_ent,), 
            initializer='zeros',
            trainable=True
        )
        self.edge_index = edge_index
        self.edge_type = edge_type
	
    
    def call(self, sub, rel, training=None):

        sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb				= sub_emb * rel_emb
        
        x = tf.matmul(obj_emb, all_ent, transpose_b=True)
        x = x + self.bias

        score = tf.sigmoid(x)
        return score
		# x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		# x += self.bias.expand_as(x)

		# score = torch.sigmoid(x)
		# return score

class CompGCN_ConvE(CompGCNBase):
    def __init__(self, edge_index, edge_type, params=None):
        super().__init__(params.num_rel, params)

        self.bn0 = tf.keras.layers.BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5)
        self.bn2 = rf.keras.layers.BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5)

        self.hidden_drop = tf.keras.layers.Dropout(self.p.hid_drop)
        self.hidden_drop = tf.keras.layers.Dropout(self.p.hid_drop2)
        self.feature_drop = tf.keras.layers.Dropout(self.p.feat_drop)

        self.m_conv1 = tf.keras.layers.Conv2D(
            filters=self.p.num_filt,
            kernel_size=(self.p.ker_sz, self.p.ker_sz),
            strides=(1,1),
            padding='valid',
            use_bias=self.p.bias,
            kernel_initializer='glorot_uniform'
        )

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.embed_dim - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = tf.keras.layers.Dense(self.p.embed_dim, activation=None)

        self.bias = self.add_weight(
            name='bias',
            shape=(self.p.num_ent,),
            initializer='zeros',
            trainable=True
        )
        
        self.edge_index = edge_index
        self.edge_type = edge_type
    
        # self.bn0		= torch.nn.BatchNorm2d(1)
        # self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
        # self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
        
        # self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
        # self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
        # self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
        # self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

        # flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
        # flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
        # self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
        # self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):

        e1_embed = tf.reshape(e1_embed, (-1, 1, self.p.embed_dim))
        rel_embed = tf.reshape(rel_embed, (-1, 1, self.p.embed_dim))
        stack_inp = tf.concat([e1_embed, rel_embed], axis=1)
        x = tf.transpose(stack_inp, perm=[0, 2, 1])
        x = tf.reshape(x, (-1, 2 * self.p.k_w, self.p.embed_dim, 1))
        return x

    def call(self, sub, rel, training=None):

        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, training)
        

        stk_inp = self.concat(sub_emb, rel_emb)
        

        x = self.bn0(stk_inp, training=training)
        x = self.m_conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.feature_drop(x, training=training)


        x = tf.reshape(x, (-1, self.flat_sz))
        x = self.fc(x)
        x = self.hidden_drop2(x, training=training)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)


        x = tf.matmul(x, all_ent, transpose_b=True)

        x = x + self.bias
        score = tf.sigmoid(x)

        return score
