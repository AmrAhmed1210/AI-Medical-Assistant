using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.AuthDTOs;
using Microsoft.Extensions.Configuration;
using Microsoft.IdentityModel.Tokens;

namespace MedicalAssistant.Services.Services
{
    public class AuthService : IAuthService
    {
        private readonly IUnitOfWork _unitOfWork;
        private readonly IConfiguration _config;
        private readonly INotificationService _notificationService;

        public AuthService(IUnitOfWork unitOfWork, IConfiguration config, INotificationService notificationService)
        {
            _unitOfWork = unitOfWork;
            _config = config;
            _notificationService = notificationService;
        }

        public async Task<AuthResponseDto> RegisterAsync(RegisterDto dto)
        {
            var email = dto.Email.ToLower().Trim();

            var existing = await _unitOfWork.Patients.GetByEmailAsync(email);
            if (existing != null)
                throw new InvalidOperationException("Email already registered.");

            var existingUser = (await _unitOfWork.Repository<User>()
                .FindAsync(u => u.Email.ToLower() == email))
                .FirstOrDefault();

            if (existingUser != null)
            {
                if (existingUser.IsDeleted)
                    throw new InvalidOperationException("This account no longer exists.");

                throw new InvalidOperationException("Email already registered.");
            }

            var hashedPassword = BCrypt.Net.BCrypt.HashPassword(dto.Password);

            // Normalize DateOfBirth to UTC to satisfy Npgsql's timestamp with time zone requirement
            DateTime? birthDateUtc = dto.DateOfBirth.HasValue
                ? DateTime.SpecifyKind(dto.DateOfBirth.Value, DateTimeKind.Utc)
                : null;

            var refreshToken = GenerateRefreshToken();
            var user = new User
            {
                FullName = dto.FullName.Trim(),
                Email = email,
                PasswordHash = hashedPassword,
                Role = "Patient",
                PhoneNumber = dto.PhoneNumber?.Trim(),
                BirthDate = birthDateUtc,
                CreatedAt = DateTime.UtcNow,
                UpdatedAt = DateTime.UtcNow,
                IsActive = true,
                RefreshToken = refreshToken,
                RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7)
            };
            await _unitOfWork.Repository<User>().AddAsync(user);
            await _unitOfWork.SaveChangesAsync();

            var patient = new Patient
            {
                FullName    = dto.FullName.Trim(),
                Email       = email,
                PhoneNumber = dto.PhoneNumber?.Trim().Length > 0 ? dto.PhoneNumber.Trim() : "N/A",
                PasswordHash = hashedPassword,
                DateOfBirth = birthDateUtc ?? DateTime.UtcNow.AddYears(-25),
                Gender      = dto.Gender ?? "N/A",
                BloodType   = dto.BloodType ?? "N/A",
                IsActive    = true,
                CreatedAt   = DateTime.UtcNow,
                UserId      = user.Id
            };

            await _unitOfWork.Patients.AddAsync(patient);
            await _unitOfWork.SaveChangesAsync();

            // Create Medical Profile
            var medicalProfile = new MedicalProfile
            {
                PatientId = patient.Id,
                BloodType = dto.BloodType,
                WeightKg = dto.Weight,
                HeightCm = dto.Height,
                IsSmoker = dto.SmokingStatus != "Non-Smoker",
                SmokingDetails = dto.SmokingStatus,
                CreatedAt = DateTime.UtcNow,
                UpdatedAt = DateTime.UtcNow
            };
            await _unitOfWork.Repository<MedicalProfile>().AddAsync(medicalProfile);
            await _unitOfWork.SaveChangesAsync();

            await _notificationService.NotifyNewUserRegistered(
                patient.Id, patient.FullName, patient.Email, "Patient");

            var token = GenerateToken(patient.FullName, patient.Email, "Patient", user.Id.ToString(), patient.Id.ToString());

            return new AuthResponseDto
            {
                AccessToken = token,
                RefreshToken = refreshToken,
                ExpiresIn = 86400,
                User = new UserDto
                {
                    Id = user.Id,
                    FullName = patient.FullName,
                    Email = patient.Email,
                    Role = "Patient",
                    PhotoUrl = patient.ImageUrl
                }
            };
        }

        public async Task<AuthResponseDto> LoginAsync(LoginDto dto)
        {
            var email = dto.Email.ToLower().Trim();

            var patient = await _unitOfWork.Patients.GetByEmailAsync(email);
            if (patient != null)
            {
                if (!patient.IsActive)
                    throw new UnauthorizedAccessException("Your account has been deactivated. Please contact administration at support@yourapp.com.");

                var isVerified = BCrypt.Net.BCrypt.Verify(dto.Password, patient.PasswordHash);
                if (!isVerified) throw new UnauthorizedAccessException("Invalid email or password.");

                var patientUser = (await _unitOfWork.Repository<User>().FindAsync(u => u.Email == email)).FirstOrDefault();
                if (patientUser == null)
                {
                    patientUser = new User
                    {
                        FullName = patient.FullName,
                        Email = patient.Email,
                        PasswordHash = patient.PasswordHash,
                        Role = "Patient",
                        PhoneNumber = patient.PhoneNumber,
                        BirthDate = patient.DateOfBirth.Kind == DateTimeKind.Utc 
                            ? patient.DateOfBirth 
                            : DateTime.SpecifyKind(patient.DateOfBirth, DateTimeKind.Utc),
                        PhotoUrl = patient.ImageUrl,
                        CreatedAt = patient.CreatedAt.Kind == DateTimeKind.Utc 
                            ? patient.CreatedAt 
                            : DateTime.SpecifyKind(patient.CreatedAt, DateTimeKind.Utc),
                        IsActive = true
                    };
                    await _unitOfWork.Repository<User>().AddAsync(patientUser);
                    await _unitOfWork.SaveChangesAsync();
                }

                if (patient.UserId == null || patient.UserId != patientUser.Id)
                {
                    patient.UserId = patientUser.Id;
                    _unitOfWork.Patients.Update(patient);

                    var sessions = await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.SessionsModule.Session>()
                        .FindAsync(s => s.UserId == patient.Id);
                    foreach (var s in sessions)
                    {
                        s.UserId = patientUser.Id;
                        _unitOfWork.Repository<MedicalAssistant.Domain.Entities.SessionsModule.Session>().Update(s);
                    }

                    await _unitOfWork.SaveChangesAsync();
                }
                else
                {
                    var sessions = await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.SessionsModule.Session>()
                        .FindAsync(s => s.UserId == patient.Id);
                    if (sessions.Any())
                    {
                        foreach (var s in sessions)
                        {
                            s.UserId = patientUser.Id;
                            _unitOfWork.Repository<MedicalAssistant.Domain.Entities.SessionsModule.Session>().Update(s);
                        }
                        await _unitOfWork.SaveChangesAsync();
                    }
                }

                var refreshToken = GenerateRefreshToken();
                patientUser.RefreshToken = refreshToken;
                patientUser.RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7);
                _unitOfWork.Repository<User>().Update(patientUser);
                await _unitOfWork.SaveChangesAsync();

                return new AuthResponseDto
                {
                    AccessToken = GenerateToken(patient.FullName, patient.Email, "Patient", patientUser.Id.ToString(), patient.Id.ToString()),
                    RefreshToken = refreshToken,
                    ExpiresIn = 86400,
                    User = new UserDto
                    {
                        Id = patientUser.Id,
                        FullName = patient.FullName,
                        Email = patient.Email,
                        Role = "Patient",
                        PhotoUrl = patient.ImageUrl
                    }
                };
            }

            var users = await _unitOfWork.Repository<User>()
                .FindAsync(u => u.Email.ToLower() == email);

            var user = users.FirstOrDefault();

            if (user == null)
                throw new UnauthorizedAccessException("Invalid email or password.");

            if (string.Equals(user.Email, "admin@admin.com", StringComparison.OrdinalIgnoreCase) && 
                BCrypt.Net.BCrypt.Verify(dto.Password, user.PasswordHash))
            {
                if (!user.IsActive || user.IsDeleted)
                {
                    user.IsActive = true;
                    user.IsDeleted = false;
                    _unitOfWork.Repository<User>().Update(user);
                    await _unitOfWork.SaveChangesAsync();
                }
            }

            if (user.IsDeleted)
                throw new UnauthorizedAccessException("This account no longer exists.");

            if (!user.IsActive)
                throw new UnauthorizedAccessException("Your account has been deactivated. Please contact administration at support@yourapp.com.");

            var isUserValid = BCrypt.Net.BCrypt.Verify(dto.Password, user.PasswordHash);
            if (!isUserValid) throw new UnauthorizedAccessException("Invalid email or password.");

            string? doctorId = null;
            if (string.Equals(user.Role, "Doctor", StringComparison.OrdinalIgnoreCase))
            {
                var doc = await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.DoctorsModule.Doctor>()
                    .FindAsync(d => d.UserId == user.Id);
                doctorId = doc.FirstOrDefault()?.Id.ToString();
            }
            else if (string.Equals(user.Role, "Secretary", StringComparison.OrdinalIgnoreCase))
            {
                var sec = await _unitOfWork.Repository<MedicalAssistant.Domain.Entities.DoctorsModule.Secretary>()
                    .FindAsync(s => s.UserId == user.Id);
                doctorId = sec.FirstOrDefault()?.DoctorId.ToString();
            }

            var refreshToken = GenerateRefreshToken();
            user.RefreshToken = refreshToken;
            user.RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7);
            _unitOfWork.Repository<User>().Update(user);
            await _unitOfWork.SaveChangesAsync();

            return new AuthResponseDto
            {
                AccessToken = GenerateToken(user.FullName, user.Email, user.Role, user.Id.ToString(), null, doctorId),
                RefreshToken = refreshToken,
                ExpiresIn = 86400,
                User = new UserDto
                {
                    Id = user.Id,
                    FullName = user.FullName,
                    Email = user.Email,
                    Role = user.Role,
                    PhotoUrl = user.PhotoUrl
                }
            };
        }

        private string GenerateToken(string name, string email, string role, string userId, string? patientId = null, string? doctorId = null)
        {
            var key    = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_config["Jwt:Key"]!));
            var creds  = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);
            var expiry = DateTime.UtcNow.AddDays(int.Parse(_config["Jwt:ExpiresInDays"] ?? "7"));

            var claimsList = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, userId),
                new Claim("UserId", userId),
                new Claim("name",   name),
                new Claim(ClaimTypes.Email, email),
                new Claim(ClaimTypes.Role,  role),
            };

            if (!string.IsNullOrEmpty(patientId))
            {
                claimsList.Add(new Claim("PatientId", patientId));
            }

            if (!string.IsNullOrEmpty(doctorId))
            {
                claimsList.Add(new Claim("DoctorId", doctorId));
            }

            var token = new JwtSecurityToken(
                issuer:             _config["Jwt:Issuer"],
                audience:           _config["Jwt:Audience"],
                claims:             claimsList,
                expires:            expiry,
                signingCredentials: creds
            );

            return new JwtSecurityTokenHandler().WriteToken(token);
        }

        public async Task<AuthResponseDto> RefreshTokenAsync(RefreshTokenRequestDto dto)
        {
            var principal = GetPrincipalFromExpiredToken(dto.AccessToken);
            if (principal == null)
                throw new SecurityTokenException("Invalid access token");

            var userIdClaim = principal.FindFirst("UserId") ?? principal.FindFirst(ClaimTypes.NameIdentifier);
            if (userIdClaim == null)
                throw new SecurityTokenException("Invalid token claims");

            var userId = int.Parse(userIdClaim.Value);
            var user = await _unitOfWork.Repository<User>().GetByIdAsync(userId);

            if (user == null || user.RefreshToken != dto.RefreshToken || user.RefreshTokenExpiryTime <= DateTime.UtcNow)
                throw new SecurityTokenException("Invalid refresh token");

            var patientId = principal.FindFirst("PatientId")?.Value;
            var doctorId = principal.FindFirst("DoctorId")?.Value;

            var newAccessToken = GenerateToken(user.FullName, user.Email, user.Role, user.Id.ToString(), patientId, doctorId);
            var newRefreshToken = GenerateRefreshToken();

            user.RefreshToken = newRefreshToken;
            user.RefreshTokenExpiryTime = DateTime.UtcNow.AddDays(7);
            _unitOfWork.Repository<User>().Update(user);
            await _unitOfWork.SaveChangesAsync();

            return new AuthResponseDto
            {
                AccessToken = newAccessToken,
                RefreshToken = newRefreshToken,
                ExpiresIn = 86400,
                User = new UserDto
                {
                    Id = user.Id,
                    FullName = user.FullName,
                    Email = user.Email,
                    Role = user.Role,
                    PhotoUrl = user.PhotoUrl
                }
            };
        }

        private static string GenerateRefreshToken()
        {
            var randomNumber = new byte[64];
            using var rng = System.Security.Cryptography.RandomNumberGenerator.Create();
            rng.GetBytes(randomNumber);
            return Convert.ToBase64String(randomNumber);
        }

        private ClaimsPrincipal? GetPrincipalFromExpiredToken(string? token)
        {
            var tokenValidationParameters = new TokenValidationParameters
            {
                ValidateAudience = false,
                ValidateIssuer = false,
                ValidateIssuerSigningKey = true,
                IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_config["Jwt:Key"]!)),
                ValidateLifetime = false
            };

            var tokenHandler = new JwtSecurityTokenHandler();
            var principal = tokenHandler.ValidateToken(token, tokenValidationParameters, out SecurityToken securityToken);
            if (securityToken is not JwtSecurityToken jwtSecurityToken || !jwtSecurityToken.Header.Alg.Equals(SecurityAlgorithms.HmacSha256, StringComparison.InvariantCultureIgnoreCase))
                throw new SecurityTokenException("Invalid token");

            return principal;
        }
    }
}
