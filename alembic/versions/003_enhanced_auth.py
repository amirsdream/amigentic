"""Enhanced authentication system with profiles, sessions, and analytics

Revision ID: 003_enhanced_auth
Revises: 002_chat_sessions
Create Date: 2024-12-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_enhanced_auth'
down_revision: Union[str, None] = '002_chat_sessions'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns to user_profiles for enhanced auth
    with op.batch_alter_table('user_profiles') as batch_op:
        # Email (optional)
        batch_op.add_column(sa.Column('email', sa.String(255), nullable=True))
        
        # Account status and role
        batch_op.add_column(sa.Column('role', sa.String(50), nullable=True, server_default='user'))
        batch_op.add_column(sa.Column('status', sa.String(50), nullable=True, server_default='active'))
        
        # Location tracking
        batch_op.add_column(sa.Column('timezone', sa.String(100), nullable=True))
        batch_op.add_column(sa.Column('country', sa.String(100), nullable=True))
        batch_op.add_column(sa.Column('city', sa.String(100), nullable=True))
        batch_op.add_column(sa.Column('last_ip', sa.String(45), nullable=True))
        
        # Additional profile fields
        batch_op.add_column(sa.Column('bio', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('organization', sa.String(255), nullable=True))
        
        # Timestamps
        batch_op.add_column(sa.Column('last_active_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('email_verified_at', sa.DateTime(), nullable=True))
        
        # Limits
        batch_op.add_column(sa.Column('daily_query_limit', sa.Integer(), nullable=True, server_default='100'))
        batch_op.add_column(sa.Column('monthly_token_limit', sa.Integer(), nullable=True, server_default='1000000'))
    
    # Create API keys table
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('key_hash', sa.String(255), nullable=False),
        sa.Column('key_prefix', sa.String(10), nullable=False),
        sa.Column('scopes', sa.JSON(), nullable=True),
        sa.Column('rate_limit', sa.Integer(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('last_used_ip', sa.String(45), nullable=True),
        sa.Column('usage_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('is_active', sa.Boolean(), nullable=True, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user_profiles.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_api_keys_user_id', 'api_keys', ['user_id'], unique=False)
    op.create_index('ix_api_keys_key_prefix', 'api_keys', ['key_prefix'], unique=False)
    
    # Create user sessions table
    op.create_table(
        'user_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(255), nullable=False),
        sa.Column('refresh_token_hash', sa.String(255), nullable=True),
        sa.Column('device_info', sa.JSON(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('location', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('last_activity_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, server_default='1'),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('revoke_reason', sa.String(255), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user_profiles.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_user_sessions_user_id', 'user_sessions', ['user_id'], unique=False)
    op.create_index('ix_user_sessions_session_id', 'user_sessions', ['session_id'], unique=True)
    
    # Create usage logs table (for data lake)
    op.create_table(
        'usage_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('action', sa.String(255), nullable=True),
        sa.Column('query', sa.Text(), nullable=True),
        sa.Column('query_hash', sa.String(64), nullable=True),
        sa.Column('request_id', sa.String(255), nullable=True),
        sa.Column('session_id', sa.String(255), nullable=True),
        sa.Column('model_used', sa.String(100), nullable=True),
        sa.Column('agents_used', sa.JSON(), nullable=True),
        sa.Column('tools_used', sa.JSON(), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('output_tokens', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('total_tokens', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('estimated_cost', sa.Float(), nullable=True, server_default='0'),
        sa.Column('duration_ms', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('status', sa.String(50), nullable=True, server_default='success'),
        sa.Column('error_code', sa.String(100), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user_profiles.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_usage_logs_user_id', 'usage_logs', ['user_id'], unique=False)
    op.create_index('ix_usage_logs_event_type', 'usage_logs', ['event_type'], unique=False)
    op.create_index('ix_usage_logs_created_at', 'usage_logs', ['created_at'], unique=False)
    op.create_index('ix_usage_logs_request_id', 'usage_logs', ['request_id'], unique=False)
    
    # Create audit logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(100), nullable=True),
        sa.Column('resource_id', sa.String(255), nullable=True),
        sa.Column('old_values', sa.JSON(), nullable=True),
        sa.Column('new_values', sa.JSON(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('session_id', sa.String(255), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=True, server_default='1'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user_profiles.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_audit_logs_user_id', 'audit_logs', ['user_id'], unique=False)
    op.create_index('ix_audit_logs_action', 'audit_logs', ['action'], unique=False)
    op.create_index('ix_audit_logs_created_at', 'audit_logs', ['created_at'], unique=False)
    
    # Create user preferences table
    op.create_table(
        'user_preferences',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('notification_email', sa.Boolean(), nullable=True, server_default='1'),
        sa.Column('notification_push', sa.Boolean(), nullable=True, server_default='1'),
        sa.Column('weekly_digest', sa.Boolean(), nullable=True, server_default='0'),
        sa.Column('default_model', sa.String(100), nullable=True),
        sa.Column('auto_save_conversations', sa.Boolean(), nullable=True, server_default='1'),
        sa.Column('show_token_usage', sa.Boolean(), nullable=True, server_default='1'),
        sa.Column('compact_mode', sa.Boolean(), nullable=True, server_default='0'),
        sa.Column('keyboard_shortcuts', sa.Boolean(), nullable=True, server_default='1'),
        sa.Column('language', sa.String(10), nullable=True, server_default='en'),
        sa.Column('custom_instructions', sa.Text(), nullable=True),
        sa.Column('favorite_agents', sa.JSON(), nullable=True),
        sa.Column('pinned_conversations', sa.JSON(), nullable=True),
        sa.Column('ui_settings', sa.JSON(), nullable=True),
        sa.Column('api_settings', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user_profiles.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_user_preferences_user_id', 'user_preferences', ['user_id'], unique=True)
    
    # Create user analytics table (aggregated for performance)
    op.create_table(
        'user_analytics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('period_start', sa.DateTime(), nullable=False),
        sa.Column('period_end', sa.DateTime(), nullable=False),
        sa.Column('period_type', sa.String(20), nullable=False),
        sa.Column('total_queries', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('total_tokens', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('total_cost', sa.Float(), nullable=True, server_default='0'),
        sa.Column('unique_sessions', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('avg_response_time_ms', sa.Float(), nullable=True, server_default='0'),
        sa.Column('error_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('models_used', sa.JSON(), nullable=True),
        sa.Column('agents_used', sa.JSON(), nullable=True),
        sa.Column('tools_used', sa.JSON(), nullable=True),
        sa.Column('peak_hour', sa.Integer(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user_profiles.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_user_analytics_user_id', 'user_analytics', ['user_id'], unique=False)
    op.create_index('ix_user_analytics_period', 'user_analytics', ['period_start', 'period_end'], unique=False)


def downgrade() -> None:
    # Drop new tables
    op.drop_table('user_analytics')
    op.drop_table('user_preferences')
    op.drop_table('audit_logs')
    op.drop_table('usage_logs')
    op.drop_table('user_sessions')
    op.drop_table('api_keys')
    
    # Remove added columns from user_profiles
    with op.batch_alter_table('user_profiles') as batch_op:
        batch_op.drop_column('monthly_token_limit')
        batch_op.drop_column('daily_query_limit')
        batch_op.drop_column('email_verified_at')
        batch_op.drop_column('last_active_at')
        batch_op.drop_column('organization')
        batch_op.drop_column('bio')
        batch_op.drop_column('last_ip')
        batch_op.drop_column('city')
        batch_op.drop_column('country')
        batch_op.drop_column('timezone')
        batch_op.drop_column('status')
        batch_op.drop_column('role')
        batch_op.drop_column('email')
