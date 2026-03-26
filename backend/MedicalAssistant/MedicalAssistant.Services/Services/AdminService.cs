using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Admin;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Domain.Entities.PatientModule;

namespace MedicalAssistant.Services.Services;

public class AdminService : IAdminService
{
    private readonly IUnitOfWork _unitOfWork;

    public AdminService(IUnitOfWork unitOfWork)
    {
        _unitOfWork = unitOfWork;
    }

    public async Task<SystemStatsDto> GetSystemStatsAsync()
    {
        var usersCount = await _unitOfWork.Repository<User>().CountAsync();
        var doctorsCount = await _unitOfWork.Repository<Doctor>().CountAsync();
        var patientsCount = await _unitOfWork.Repository<Patient>().CountAsync();

        return new SystemStatsDto
        {
            TotalUsers = usersCount,
            TotalDoctors = doctorsCount,
            TotalPatients = patientsCount,
            TotalAppointments = 0 
        };
    }

    public async Task<IEnumerable<UserManagementDto>> GetAllUsersAsync()
    {
        var users = await _unitOfWork.Repository<User>().GetAllAsync();
        return users.Select(u => new UserManagementDto {
    
            Id = u.Id, 
            Name = u.FullName,
            Email = u.Email,
            Role = u.Role,
            IsActive = u.IsActive
        });
    }

    public async Task<bool> ToggleUserStatusAsync(int userId)
    {
        var users = await _unitOfWork.Repository<User>().GetAllAsync();
      
        var user = users.FirstOrDefault(u => u.Id == userId);
        
        if (user == null) return false;
        
        user.IsActive = !user.IsActive;

        await _unitOfWork.SaveChangesAsync(); 
        
        return true;
    }
}