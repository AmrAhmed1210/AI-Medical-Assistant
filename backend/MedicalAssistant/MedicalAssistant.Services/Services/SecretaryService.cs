using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Secretary;

namespace MedicalAssistant.Services.Services;

public class SecretaryService : ISecretaryService
{
    private readonly IUnitOfWork _unitOfWork;

    public SecretaryService(IUnitOfWork unitOfWork)
    {
        _unitOfWork = unitOfWork;
    }

    public async Task<SecretaryDto> AddSecretaryAsync(int doctorUserId, CreateSecretaryDto dto)
    {
        // 1. Get Doctor from UserId
        var doctor = (await _unitOfWork.Repository<Doctor>().FindAsync(d => d.UserId == doctorUserId)).FirstOrDefault();
        if (doctor == null) throw new UnauthorizedAccessException("Doctor not found.");

        // 2. Check if email exists
        var email = dto.Email.ToLower().Trim();
        var existingUser = (await _unitOfWork.Repository<User>().FindAsync(u => u.Email.ToLower() == email)).FirstOrDefault();
        if (existingUser != null) throw new InvalidOperationException("Email already registered.");

        // 3. Create User record
        var hashedPassword = BCrypt.Net.BCrypt.HashPassword(dto.Password);
        var user = new User
        {
            FullName = dto.FullName.Trim(),
            Email = email,
            PasswordHash = hashedPassword,
            Role = "Secretary",
            CreatedAt = DateTime.UtcNow,
            IsActive = true
        };
        await _unitOfWork.Repository<User>().AddAsync(user);
        await _unitOfWork.SaveChangesAsync();

        // 4. Create Secretary record
        var secretary = new Secretary
        {
            UserId = user.Id,
            DoctorId = doctor.Id,
            FullName = dto.FullName.Trim(),
            CreatedAt = DateTime.UtcNow,
            IsActive = true
        };
        await _unitOfWork.Repository<Secretary>().AddAsync(secretary);
        await _unitOfWork.SaveChangesAsync();

        return new SecretaryDto(secretary.Id, user.Id, doctor.Id, secretary.FullName, user.Email, secretary.IsActive);
    }

    public async Task<IEnumerable<SecretaryDto>> GetSecretariesForDoctorAsync(int doctorUserId)
    {
        var doctor = (await _unitOfWork.Repository<Doctor>().FindAsync(d => d.UserId == doctorUserId)).FirstOrDefault();
        if (doctor == null) return Enumerable.Empty<SecretaryDto>();

        var secretaries = await _unitOfWork.Repository<Secretary>().FindAsync(s => s.DoctorId == doctor.Id);
        var results = new List<SecretaryDto>();

        foreach (var s in secretaries)
        {
            // Direct query to avoid any tracking/cache issues with GetByIdAsync
            var user = (await _unitOfWork.Repository<User>().FindAsync(u => u.Id == s.UserId)).FirstOrDefault();
            results.Add(new SecretaryDto(s.Id, s.UserId, s.DoctorId, s.FullName, user?.Email ?? "No Email Found", s.IsActive));
        }

        return results;
    }

    public async Task<bool> DeleteSecretaryAsync(int doctorUserId, int secretaryId)
    {
        var doctor = (await _unitOfWork.Repository<Doctor>().FindAsync(d => d.UserId == doctorUserId)).FirstOrDefault();
        if (doctor == null) return false;

        var secretary = await _unitOfWork.Repository<Secretary>().GetByIdAsync(secretaryId);
        if (secretary == null || secretary.DoctorId != doctor.Id) return false;

        // Optionally delete the user too or just deactivate
        var user = await _unitOfWork.Repository<User>().GetByIdAsync(secretary.UserId);
        if (user != null)
        {
            user.IsDeleted = true;
            _unitOfWork.Repository<User>().Update(user);
        }

        _unitOfWork.Repository<Secretary>().Delete(secretary);
        await _unitOfWork.SaveChangesAsync();
        return true;
    }

    public async Task<int?> GetDoctorIdForSecretaryAsync(int secretaryUserId)
    {
        var secretary = (await _unitOfWork.Repository<Secretary>().FindAsync(s => s.UserId == secretaryUserId)).FirstOrDefault();
        return secretary?.DoctorId;
    }
}
