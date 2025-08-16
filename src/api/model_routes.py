"""
Model management API routes for LLM selection.
"""
import logging
from flask import Blueprint, request, jsonify
from flask_login import login_required

logger = logging.getLogger(__name__)


def create_model_blueprint(base_path: str):
    """
    Create and configure the model management blueprint.
    
    Args:
        base_path: Base path for routes (e.g., '/meetingsai')
    
    Returns:
        Blueprint: Configured Flask blueprint
    """
    # Import here to avoid circular imports
    from meeting_processor import (
        AVAILABLE_MODELS, 
        get_current_model_name, 
        set_current_model,
        get_current_model_config
    )
    
    model_bp = Blueprint('model', __name__, url_prefix=f'{base_path}/api/model')
    
    @model_bp.route('/available', methods=['GET'])
    def get_available_models():
        """Get list of available models"""
        try:
            models = []
            for model_id, config in AVAILABLE_MODELS.items():
                models.append({
                    'id': model_id,
                    'name': config['name'],
                    'description': config['description']
                })
            
            return jsonify({
                'success': True,
                'models': models,
                'current_model': get_current_model_name()
            })
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get available models'
            }), 500
    
    @model_bp.route('/current', methods=['GET'])
    def get_current_model():
        """Get current active model"""
        try:
            current_model = get_current_model_name()
            current_config = get_current_model_config()
            
            return jsonify({
                'success': True,
                'current_model': current_model,
                'model_info': {
                    'id': current_model,
                    'name': current_config['name'],
                    'description': current_config['description']
                }
            })
        except Exception as e:
            logger.error(f"Error getting current model: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to get current model'
            }), 500
    
    @model_bp.route('/switch', methods=['POST'])
    def switch_model():
        """Switch to a different model"""
        try:
            data = request.get_json()
            if not data or 'model_id' not in data:
                return jsonify({
                    'success': False,
                    'error': 'model_id is required'
                }), 400
            
            model_id = data['model_id']
            
            # Attempt to switch model
            success = set_current_model(model_id)
            
            if success:
                new_config = get_current_model_config()
                logger.info(f"Model switched to: {model_id}")
                return jsonify({
                    'success': True,
                    'message': f'Successfully switched to {new_config["name"]}',
                    'current_model': model_id,
                    'model_info': {
                        'id': model_id,
                        'name': new_config['name'],
                        'description': new_config['description']
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Failed to switch to model: {model_id}'
                }), 500
                
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return jsonify({
                'success': False,
                'error': 'Failed to switch model'
            }), 500
    
    return model_bp