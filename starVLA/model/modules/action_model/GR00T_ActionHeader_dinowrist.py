import torch

from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead


class FlowmatchingActionHeadDinoWrist(FlowmatchingActionHead):
    @staticmethod
    def _concat_prefix_features(state_features, wrist_features):
        if state_features is None:
            return wrist_features
        if wrist_features is None:
            return state_features
        return torch.cat((state_features, wrist_features), dim=1)

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        wrist_features: torch.Tensor = None,
    ):
        device = vl_embs.device

        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = self.state_encoder(state) if state is not None else None
        prefix_features = self._concat_prefix_features(state_features, wrist_features)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((prefix_features, future_tokens, action_features), dim=1)
            if prefix_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]
        return ((pred_actions - velocity) ** 2).mean()

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        wrist_features: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None
        prefix_features = self._concat_prefix_features(state_features, wrist_features)

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = (
                torch.cat((prefix_features, future_tokens, action_features), dim=1)
                if prefix_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity

        return actions


def get_action_model(config=None):
    return FlowmatchingActionHeadDinoWrist(full_config=config)
