ARG BASE_IMAGE="ghcr.io/coreweave/ml-containers/torch:bb02bee-base-cuda11.8.0-torch2.0.0-vision0.15.1-audio2.0.1"

FROM ${BASE_IMAGE}

RUN apt-get -qq update && \
    DEBIAN_FRONTEND=noninteractive \
      apt-get -qq install --no-install-recommends -y \
      # Critical packages:
      ssh ca-certificates tini bash \
      # Helpful packages:
      libncurses5 curl wget sudo htop git rsync \
      tmux unzip nano vim apt-utils iputils-ping && \
    apt-get clean && \
    # Wipe the server-side SSH keys on the container image level
    # to prevent leaking the private host keys, which could
    # potentially allow impersonation of the SSH server by an attacker.
    rm /etc/ssh/ssh_host_*

# Since there are no host keys, the SSH server
# MUST be configured at runtime by running:
#   dpkg-reconfigure openssh-server
# (Or by adding custom host key files to /etc/ssh/) before launching it with:
#   service ssh start
# Or (blocking):
#   service ssh start -D

RUN \
    # Configure the privilege separation directory for sshd
    # See here for details: https://github.com/openssh/openssh-portable/blob/master/README.privsep
    install -d --mode=0755 --owner=0 --group=0 /var/run/sshd && \
    # Configure an empty authorized keys file with correct permissions
    install -d --mode=0700 --owner=0 --group=0 /root/.ssh && \
    install --mode=600 --owner=0 --group=0 /dev/null /root/.ssh/authorized_keys && \
    # Allow only public key authentication
    install --mode=600 --owner=0 --group=0 /dev/null /etc/ssh/sshd_config.d/10-key-auth.conf && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config.d/10-key-auth.conf && \
    echo "PermitRootLogin without-password" >> /etc/ssh/sshd_config.d/10-key-auth.conf && \
    # Prevent the user from being kicked off after login
    # See here for details: https://stackoverflow.com/questions/21391142
    sed -i -E -e \
      's:session(\s*)required(\s*)pam_loginuid\.so:session\1optional\2pam_loginuid.so:g' \
      /etc/pam.d/sshd && \
    # Fix sudo bug: https://github.com/sudo-project/sudo/issues/42
    echo 'Set disable_coredump false' >> /etc/sudo.conf

RUN chsh -s /bin/bash root

EXPOSE 22
