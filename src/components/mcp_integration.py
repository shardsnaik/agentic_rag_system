from typing import TypedDict, List, Dict, Optional, Any
import os, json
from datetime import datetime
from pathlib import Path


# ============================================================================
# MCP SERVER INTEGRATION
# ============================================================================

class MCPServer:
    """Model Context Protocol server for file access"""
    
    def __init__(self):
        self.local_file_system = LocalFileSystemMCP()
        # self.google_drive = GoogleDriveMCP()
    
    def list_local_files(self, directory: str) -> List[Dict[str, Any]]:
        """List files in local directory via MCP"""
        return self.local_file_system.list_files(directory)
    
    def read_local_file(self, filepath: str) -> str:
        """Read local file via MCP"""
        return self.local_file_system.read_file(filepath)
    
    # def list_drive_files(self, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
    #     """List files from Google Drive via MCP"""
    #     return self.google_drive.list_files(folder_id)
    
    # def download_drive_file(self, file_id: str, save_path: str) -> str:
    #     """Download file from Google Drive via MCP"""
    #     return self.google_drive.download_file(file_id, save_path)


class LocalFileSystemMCP:
    """MCP interface for local file system"""
    
    def list_files(self, directory: str = ".") -> List[Dict[str, Any]]:
        """List files in directory"""
        files = []
        path = Path(directory)
        
        if not path.exists():
            return files
        
        for item in path.rglob("*"):
            if item.is_file():
                files.append({
                    "name": item.name,
                    "path": str(item),
                    "size": item.stat().st_size,
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    "type": item.suffix
                })
        
        return files
    
    def read_file(self, filepath: str) -> str:
        """Read file content"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()


class GoogleDriveMCP:
    """MCP interface for Google Drive"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        self.credentials_path = credentials_path or os.getenv("GOOGLE_CREDENTIALS_PATH")
        self.service = None
        
        if self.credentials_path and os.path.exists(self.credentials_path):
            self._initialize_drive()
    
    def _initialize_drive(self):
        """Initialize Google Drive API"""
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
            import pickle
            
            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            creds = None
            
            # Token file
            token_path = 'token.pickle'
            
            if os.path.exists(token_path):
                with open(token_path, 'rb') as token:
                    creds = pickle.load(token)
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES)
                    creds = flow.run_local_server(port=8080)
                
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
            
            self.service = build('drive', 'v3', credentials=creds)
            print("✓ Google Drive connected via MCP")
            
        except Exception as e:
            print(f"⚠️  Google Drive MCP not available: {e}")
            self.service = None
    
    def list_files(self, folder_id: Optional[str] = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """List files from Google Drive"""
        if not self.service:
            return []
        
        try:
            query = f"'{folder_id}' in parents" if folder_id else None
            
            results = self.service.files().list(
                q=query,
                pageSize=max_results,
                fields="files(id, name, mimeType, size, modifiedTime)"
            ).execute()
            
            files = results.get('files', [])
            
            return [{
                "id": f["id"],
                "name": f["name"],
                "type": f["mimeType"],
                "size": f.get("size", 0),
                "modified": f["modifiedTime"]
            } for f in files]
            
        except Exception as e:
            print(f"Error listing Drive files: {e}")
            return []
    
    def download_file(self, file_id: str, save_path: str) -> str:
        """Download file from Google Drive"""
        if not self.service:
            raise Exception("Google Drive not initialized")
        
        try:
            from googleapiclient.http import MediaIoBaseDownload
            import io
            
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            with open(save_path, 'wb') as f:
                f.write(fh.getvalue())
            
            print(f"✓ Downloaded to {save_path}")
            return save_path
            
        except Exception as e:
            raise Exception(f"Failed to download file: {e}")
