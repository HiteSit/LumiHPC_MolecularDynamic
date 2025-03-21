import paramiko
import os
from pathlib import Path
import time

class HPCConnection:
    """
    A class to handle SSH connections to HPC systems with support for
    password-protected SSH keys.
    """
    
    def __init__(
        self,
        hostname = "lumi.csc.fi",
        username = "fuscoric",
        key_path = "~/.ssh/id_ed25519",
        key_passphrase=None
        ):
        """
        Initialize the HPC connection.
        
        Args:
            hostname (str): The hostname or IP address of the HPC system
            username (str): Your username on the HPC system
            key_path (str): Path to your SSH private key file
            key_passphrase (str, optional): Passphrase for your encrypted SSH key
        """
        self.hostname = hostname
        self.username = username
        self.key_path = os.path.expanduser(key_path)  # Expand ~ to home directory
        self.key_passphrase = key_passphrase
        self.client = None
        self.sftp = None
        
    def connect(self):
        """
        Establish SSH connection to the HPC using the provided key and passphrase.
        
        Raises:
            paramiko.ssh_exception.PasswordRequiredException: If key is encrypted and no passphrase provided
            paramiko.ssh_exception.SSHException: If connection fails for other SSH-related reasons
            FileNotFoundError: If the key file doesn't exist
        """
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            # Load the key with the passphrase if provided
            if os.path.isfile(self.key_path):
                key = paramiko.Ed25519Key.from_private_key_file(
                    self.key_path,
                    password=self.key_passphrase
                )
                self.client.connect(
                    self.hostname, 
                    username=self.username, 
                    pkey=key
                )
            else:
                raise FileNotFoundError(f"SSH key not found: {self.key_path}")
                
            # Open SFTP connection for file transfers
            self.sftp = self.client.open_sftp()
            print(f"Connected to {self.hostname}")
            
        except paramiko.ssh_exception.PasswordRequiredException:
            raise Exception("Your private key is encrypted. Please provide the passphrase using the key_passphrase parameter.")
        except Exception as e:
            # Clean up if connection fails
            if self.client:
                self.client.close()
            raise e
        
    def disconnect(self):
        """Close SSH and SFTP connections."""
        if self.sftp:
            self.sftp.close()
        if self.client:
            self.client.close()
            self.client = None
            self.sftp = None
        print(f"Disconnected from {self.hostname}")
        
    def run_command(self, command, working_dir=None):
        """
        Run a command on the HPC.
        
        Args:
            command (str): The command to execute
            working_dir (str, optional): Directory to change to before executing
                                         the command
                                         
        Returns:
            tuple: (output, error, exit_status) - The command's stdout, stderr, and exit code
        """
        if not self.client:
            raise Exception("Not connected. Call connect() first.")
            
        if working_dir:
            command = f"cd {working_dir} && {command}"
            
        print(f"Running: {command}")
        stdin, stdout, stderr = self.client.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        if exit_status != 0:
            print(f"Command failed with status {exit_status}")
            print(f"Error: {error}")
        
        return output, error, exit_status
        
    def upload_file(self, local_path, remote_path):
        """
        Upload a single file to the HPC.
        
        Args:
            local_path (str): Path to the local file to upload
            remote_path (str): Destination path on the HPC
        """
        if not self.sftp:
            raise Exception("Not connected. Call connect() first.")
            
        # Expand local path if it contains ~
        local_path = os.path.expanduser(local_path)
        
        # Create remote directory if it doesn't exist
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            try:
                self.sftp.stat(remote_dir)
            except FileNotFoundError:
                self._mkdir_p(remote_dir)
                
        print(f"Uploading {local_path} to {remote_path}")
        self.sftp.put(local_path, remote_path)
    
    def _mkdir_p(self, remote_directory):
        """
        Create remote directory and parents if they don't exist
        (similar to mkdir -p).
        
        Args:
            remote_directory (str): Path to create on the remote system
        """
        dirs = [d for d in remote_directory.split('/') if d]
        current_dir = ''
        
        for d in dirs:
            current_dir += '/' + d
            try:
                self.sftp.stat(current_dir)
            except FileNotFoundError:
                print(f"Creating remote directory: {current_dir}")
                self.sftp.mkdir(current_dir)
    
    def download_file(self, remote_path, local_path):
        """
        Download a single file from the HPC.
        
        Args:
            remote_path (str): Path to the remote file to download
            local_path (str): Destination path on the local system
        """
        if not self.sftp:
            raise Exception("Not connected. Call connect() first.")
            
        # Expand local path if it contains ~
        local_path = os.path.expanduser(local_path)
        
        # Create local directory if it doesn't exist
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
            
        print(f"Downloading {remote_path} to {local_path}")
        self.sftp.get(remote_path, local_path)
    
    def upload_directory(self, local_dir, remote_dir):
        """
        Upload a local directory to the HPC.
        
        Args:
            local_dir (str): Path to the local directory to upload
            remote_dir (str): Destination parent directory on the HPC
        """
        if not self.sftp:
            raise Exception("Not connected. Call connect() first.")
            
        # Normalize paths and handle trailing slashes
        local_dir = os.path.normpath(os.path.expanduser(local_dir))
        remote_dir = os.path.normpath(remote_dir)
        
        # Get the local directory name to preserve in remote path
        local_name = os.path.basename(local_dir)
        target_remote_dir = os.path.join(remote_dir, local_name)
        
        # Verify local path is a directory
        local_path = Path(local_dir)
        if not local_path.is_dir():
            raise ValueError(f"Local path is not a directory: {local_dir}")
        
        # Create target remote directory
        try:
            self.sftp.stat(target_remote_dir)
        except FileNotFoundError:
            self._mkdir_p(target_remote_dir)
            
        print(f"Uploading directory '{local_dir}' to '{target_remote_dir}'")
            
        # Upload all files and subdirectories
        for item in local_path.glob('**/*'):
            if item.is_file():
                # Get the relative path from the source directory
                relative_path = item.relative_to(local_path)
                remote_file_path = f"{target_remote_dir}/{relative_path}"
                
                # Create remote parent directories if needed
                remote_parent = os.path.dirname(remote_file_path)
                try:
                    self.sftp.stat(remote_parent)
                except FileNotFoundError:
                    self._mkdir_p(remote_parent)
                    
                print(f"Uploading {item} to {remote_file_path}")
                self.sftp.put(str(item), remote_file_path)
        
        print(f"Directory upload complete: {local_dir} → {target_remote_dir}")
    
    def download_directory(self, remote_dir, local_dir):
        """
        Download a remote directory from the HPC.
        
        Args:
            remote_dir (str): Path to the remote directory to download
            local_dir (str): Destination parent directory on the local system
        """
        if not self.sftp:
            raise Exception("Not connected. Call connect() first.")
            
        # Normalize paths and handle trailing slashes
        remote_dir = os.path.normpath(remote_dir)
        local_dir = os.path.normpath(os.path.expanduser(local_dir))
        
        # Get the remote directory name to preserve in local path
        remote_name = os.path.basename(remote_dir)
        target_local_dir = os.path.join(local_dir, remote_name)
        
        # Create target local directory
        target_local_path = Path(target_local_dir)
        target_local_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading directory '{remote_dir}' to '{target_local_dir}'")
        
        # Get remote files with full paths (only files, not directories)
        files = self._get_all_remote_files(remote_dir)
        
        if not files:
            print(f"Warning: No files found in remote directory: {remote_dir}")
            return
        
        for remote_path in files:
            try:
                # Create proper relative paths by stripping remote_dir
                remote_path_normalized = os.path.normpath(remote_path)
                
                # Skip the directory itself
                if remote_path_normalized == remote_dir:
                    continue
                    
                # Get the path relative to the remote directory
                if remote_path_normalized.startswith(remote_dir):
                    # Get relative path (remove remote_dir prefix plus separator)
                    rel_path = remote_path_normalized[len(remote_dir):]
                    rel_path = rel_path.lstrip('/').lstrip('\\')
                    local_file_path = target_local_path / rel_path
                    
                    # Create parent directories if needed
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Skip directories, only download files
                    try:
                        attr = self.sftp.stat(remote_path)
                        if self._is_dir_from_attr(attr):
                            continue
                        
                        print(f"Downloading {remote_path} to {local_file_path}")
                        self.sftp.get(remote_path, str(local_file_path))
                    except Exception as e:
                        print(f"Error downloading {remote_path}: {str(e)}")
                else:
                    print(f"Warning: Skipping file outside target directory: {remote_path}")
            except Exception as e:
                print(f"Error processing path {remote_path}: {str(e)}")
        
        print(f"Directory download complete: {remote_dir} → {target_local_dir}")
        
    def _get_all_remote_files(self, remote_dir):
        """
        Recursively get all files in a remote directory.
        
        Args:
            remote_dir (str): Remote directory path to scan
            
        Returns:
            list: List of full paths to all files in the directory and subdirectories
        """
        files = []
        try:
            remote_dir = remote_dir.rstrip('/')  # Remove trailing slash for consistency
            
            for entry in self.sftp.listdir_attr(remote_dir):
                remote_path = f"{remote_dir}/{entry.filename}"
                
                if self._is_dir_from_attr(entry):
                    # Recursively add files from subdirectories
                    files.extend(self._get_all_remote_files(remote_path))
                else:
                    # Only add files, not directories
                    files.append(remote_path)
        except FileNotFoundError:
            print(f"Remote directory not found: {remote_dir}")
        except Exception as e:
            print(f"Error accessing remote directory {remote_dir}: {str(e)}")
        return files
        
    def _is_dir_from_attr(self, attr):
        """
        Check if a remote entry is a directory using its attributes.
        
        Args:
            attr: SFTP file attribute entry
            
        Returns:
            bool: True if the entry is a directory, False otherwise
        """
        # Using the standard S_IFDIR bit mask (0o40000) to check for directory
        return attr.st_mode & 0o40000 == 0o40000
        
    def _is_dir(self, entry):
        """
        Check if a remote entry is a directory.
        Alias for _is_dir_from_attr for backward compatibility.
        
        Args:
            entry: SFTP file attribute entry
            
        Returns:
            bool: True if the entry is a directory, False otherwise
        """
        return self._is_dir_from_attr(entry)
        
    def submit_job(self, script_path, working_dir=None):
        """
        Submit a job to the HPC queue system (typically SLURM).
        
        Args:
            script_path (str): Path to the job script on the remote system
            working_dir (str, optional): Directory to run the sbatch command from
            
        Returns:
            str or None: Job ID if submission was successful, None otherwise
        """
        if working_dir:
            command = f"cd {working_dir} && sbatch {script_path}"
        else:
            command = f"sbatch {script_path}"
            
        output, error, status = self.run_command(command)
        
        if status == 0:
            # Typically sbatch outputs "Submitted batch job JOBID"
            job_id = output.strip().split()[-1]
            print(f"Job submitted with ID: {job_id}")
            return job_id
        else:
            print(f"Failed to submit job: {error}")
            return None
    
    def check_job_status(self, job_id):
        """
        Check the status of a submitted job.
        
        Args:
            job_id (str): Job ID to check
            
        Returns:
            str: Status of the job (PENDING, RUNNING, COMPLETED, FAILED, etc.)
        """
        command = f"squeue -j {job_id}"
        output, error, status = self.run_command(command)
        
        if "Invalid job id specified" in error:
            # Check if it completed successfully with sacct
            sacct_cmd = f"sacct -j {job_id} -o State -n"
            sacct_out, _, _ = self.run_command(sacct_cmd)
            
            if "COMPLETED" in sacct_out:
                return "COMPLETED"
            elif "FAILED" in sacct_out:
                return "FAILED"
            elif "CANCELLED" in sacct_out:
                return "CANCELLED"
            else:
                return "UNKNOWN"
            
        if status != 0:
            return "UNKNOWN"
            
        # Parse squeue output to determine status
        lines = output.strip().split('\n')
        if len(lines) < 2:  # No job data returned
            return "COMPLETED"
            
        # Parse status from squeue output
        job_info = lines[1].split()
        if len(job_info) >= 5:
            return job_info[4]  # Status column in squeue output
        
        return "UNKNOWN"
        
    def download_and_rename_directory(self, remote_path, local_path, new_name=None):
        """
        Download a directory from remote HPC and optionally rename it.
        
        Args:
            remote_path (str): Path to the remote directory to download
            local_path (str): Destination parent directory on local system
            new_name (str, optional): New name for the downloaded directory.
                                     If not provided, the original name is used.
        """
        # Normalize paths
        remote_path = os.path.normpath(remote_path)
        local_path = os.path.normpath(os.path.expanduser(local_path))
        
        if new_name:
            # If a new name is provided, we need to download to parent dir and rename
            # Create parent directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)
            
            # Get original files with the standard download method
            temp_local_dir = os.path.join(local_path, os.path.basename(remote_path))
            self.download_directory(remote_path, local_path)
            
            # Rename if the target directory exists
            if os.path.exists(temp_local_dir):
                target_dir = os.path.join(local_path, new_name)
                # Remove target directory if it already exists
                if os.path.exists(target_dir):
                    import shutil
                    shutil.rmtree(target_dir)
                # Rename to target name
                os.rename(temp_local_dir, target_dir)
                print(f"Renamed downloaded directory to: {target_dir}")
        else:
            # No renaming, use standard download
            self.download_directory(remote_path, local_path)
        
    def upload_and_rename_directory(self, local_path, remote_path, new_name=None):
        """
        Upload a directory to remote HPC and optionally rename it.
        
        Args:
            local_path (str): Path to local directory to upload
            remote_path (str): Destination parent directory on remote system
            new_name (str, optional): New name for the uploaded directory.
                                     If not provided, the original name is used.
        """
        # Normalize paths
        local_path = os.path.normpath(os.path.expanduser(local_path))
        remote_path = os.path.normpath(remote_path)
        
        if new_name:
            # If new name is provided, we need a special handling
            local_name = os.path.basename(local_path)
            orig_remote_dir = os.path.join(remote_path, local_name)
            target_remote_dir = os.path.join(remote_path, new_name)
            
            # First upload with original name
            self.upload_directory(local_path, remote_path)
            
            # Then rename on remote system
            try:
                # Check if target already exists and remove it
                try:
                    self.sftp.stat(target_remote_dir)
                    # If exists, remove it with a command
                    self.run_command(f"rm -rf {target_remote_dir}")
                except FileNotFoundError:
                    pass
                
                # Rename using mv command
                _, error, status = self.run_command(f"mv {orig_remote_dir} {target_remote_dir}")
                if status == 0:
                    print(f"Renamed uploaded directory to: {target_remote_dir}")
                else:
                    print(f"Failed to rename directory: {error}")
            except Exception as e:
                print(f"Error during remote rename: {str(e)}")
        else:
            # No renaming needed, use standard upload
            self.upload_directory(local_path, remote_path)