using AutoMapper;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;

namespace MedicalAssistant.Application.Services;

public class DoctorService : IDoctorService
{
    private readonly IUnitOfWork _unitOfWork;
    private readonly IMapper _mapper;

    public DoctorService(IUnitOfWork unitOfWork, IMapper mapper)
    {
        _unitOfWork = unitOfWork;
        _mapper = mapper;
    }

    public async Task<IReadOnlyList<DoctorDTO>> GetAllDoctorsAsync()
    {
        // الوصول للمستودع من خلال الـ Unit of Work
        var doctors = await _unitOfWork.Repository<Doctor>().GetAllAsync();
        return _mapper.Map<IReadOnlyList<DoctorDTO>>(doctors);
    }

    public async Task<DoctorDetailsDTO?> GetDoctorByIdAsync(int id)
    {
        // استخدام المستودع الخاص بالأطباء داخل الـ Unit of Work لجلب تفاصيل كاملة
        var doctor = await _unitOfWork.Repository<Doctor>().GetByIdAsync(id);
        return doctor is null ? null : _mapper.Map<DoctorDetailsDTO>(doctor);
    }

    public async Task<IReadOnlyList<DoctorDTO>> GetAvailableDoctorsAsync()
    {
        // ملاحظة: إذا كان لديك منطق خاص في DoctorRepository، تأكد من تسجيله في الـ Unit of Work
        var doctors = await _unitOfWork.Doctors.GetAvailableDoctorsAsync();
        return _mapper.Map<IReadOnlyList<DoctorDTO>>(doctors);
    }

    public async Task<IReadOnlyList<DoctorDTO>> GetDoctorsBySpecialtyAsync(int specialtyId)
    {
        var doctors = await _unitOfWork.Doctors.GetBySpecialtyAsync(specialtyId);
        return _mapper.Map<IReadOnlyList<DoctorDTO>>(doctors);
    }

    public async Task<IReadOnlyList<DoctorDTO>> SearchDoctorsAsync(string name)
    {
        var doctors = await _unitOfWork.Doctors.SearchByNameAsync(name);
        return _mapper.Map<IReadOnlyList<DoctorDTO>>(doctors);
    }

    public async Task<IReadOnlyList<DoctorDTO>> GetTopRatedDoctorsAsync(int count)
    {
        var doctors = await _unitOfWork.Doctors.GetTopRatedDoctorsAsync(count);
        return _mapper.Map<IReadOnlyList<DoctorDTO>>(doctors);
    }

    public async Task<(IReadOnlyList<DoctorDTO> Items, int TotalCount)> GetPaginatedDoctorsAsync(int pageNumber, int pageSize)
    {
        var (items, totalCount) = await _unitOfWork.Doctors.GetPaginatedAsync(pageNumber, pageSize);
        var mappedItems = _mapper.Map<IReadOnlyList<DoctorDTO>>(items);
        return (mappedItems, totalCount);
    }
}