import itertools
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
import tqdm
from model.cycleGAN import Generator, Discriminator
from data.dataset import ImageDataset
from utils import *


def test(test_set):
    imgs = random.choices(test_set, k=4)
    B = torch.stack([img[1] for img in imgs])
    B = B.to(device)
    fake_A = netG_B2A(B)
    visualize(B, fake_A)


device = torch.device('cuda:0')
input_channels = 3
output_channels = 3
lr = 2e-4
batch_size = 32
epochs = 200
netG_A2B = Generator(input_channels, output_channels).to(device)
netG_A2B.load_state_dict(torch.load('pic2monet/G_B2A.pkl'))
netG_B2A = Generator(output_channels, input_channels).to(device)
netG_B2A.load_state_dict(torch.load('pic2monet/G_B2A.pkl'))
netD_A = Discriminator(input_channels).to(device)
netD_B = Discriminator(output_channels).to(device)

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(epochs, epochs // 2).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(epochs, epochs // 2).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(epochs, epochs // 2).step)

target_real = torch.ones((batch_size, 1), device=device)
target_fake = torch.zeros((batch_size, 1), device=device)
transform_ = T.Compose([T.Resize(int(64 * 1.12), interpolation=Image.BICUBIC),
                        T.RandomCrop(64),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize([0.5], [0.5])])

transform_test_ = T.Compose([T.Resize(256, interpolation=Image.BICUBIC),
                             T.ToTensor(),
                             T.Normalize([0.5], [0.5])])

train_set = ImageDataset(root=r'F:\datasets\vangogh2photo', transform=transform_, mode='train')
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_set = ImageDataset(root=r'F:\datasets\vangogh2photo', transform=transform_test_, mode='test')
buffer_A = ReplayBuffer()
buffer_B = ReplayBuffer()
# for epoch in range(epochs):
#     for i, batch in tqdm.tqdm(enumerate(train_loader),
#                               total=len(train_loader),
#                               desc='Epoch_%s' % epoch):
#         real_A, real_B = batch
#         real_A, real_B = real_A.to(device), real_B.to(device)
#         ############# Optimize Generator ############
#         optimizer_G.zero_grad()
#
#         # Identity loss
#         # G_A2B(B) should equal B if real B is fed
#         same_B = netG_A2B(real_B)
#         loss_identity_B = criterion_identity(same_B, real_B) * 5.0
#         # G_B2A(A) should equal A if real A is fed
#         same_A = netG_B2A(real_A)
#         loss_identity_A = criterion_identity(same_A, real_A) * 5.0
#
#         # GAN loss
#         fake_B = netG_A2B(real_A)
#         pred_true = netD_B(fake_B)
#         loss_GAN_A2B = criterion_GAN(pred_true, target_real)
#
#         fake_A = netG_B2A(real_B)
#         pred_true = netD_A(fake_A)
#         loss_GAN_B2A = criterion_GAN(pred_true, target_real)
#
#         # Cycle loss
#         recovered_A = netG_B2A(fake_B)
#         loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0
#
#         recovered_B = netG_A2B(fake_A)
#         loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0
#
#         # Total loss
#         loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
#         loss_G.backward()
#
#         optimizer_G.step()
#
#         ############# Optimize Discriminator A ############
#         optimizer_D_A.zero_grad()
#
#         # Real loss
#         pred_real = netD_A(real_A)
#         loss_D_real = criterion_GAN(pred_real, target_real)
#
#         # Fake loss
#         fake_A = buffer_A.push_and_pop(fake_A)
#         pred_true = netD_A(fake_A.detach())
#         loss_D_fake = criterion_GAN(pred_true, target_fake)
#
#         # Total loss
#         loss_D_A = (loss_D_real + loss_D_fake) * 0.5
#         loss_D_A.backward()
#
#         optimizer_D_A.step()
#
#         ############# Optimize Discriminator B ############
#         optimizer_D_B.zero_grad()
#
#         # Real loss
#         pred_real = netD_B(real_B)
#         loss_D_real = criterion_GAN(pred_real, target_real)
#
#         # Fake loss
#         fake_B = buffer_B.push_and_pop(fake_B)
#         pred_true = netD_B(fake_B.detach())
#         loss_D_fake = criterion_GAN(pred_true, target_fake)
#
#         # Total loss
#         loss_D_B = (loss_D_real + loss_D_fake) * 0.5
#         loss_D_B.backward()
#
#         optimizer_D_B.step()
#
#     lr_scheduler_G.step()
#     lr_scheduler_D_A.step()
#     lr_scheduler_D_B.step()
#     imgs = random.choices(test_set, k=4)
#     A = torch.stack([img[0] for img in imgs])
#     B = torch.stack([img[1] for img in imgs])
#     A = A.to(device)
#     B = B.to(device)
#     fake_B = netG_A2B(A)
#     fake_A = netG_B2A(B)
#     visualize(A, fake_B)
#     visualize(B, fake_A)

